import urllib.request
import json
import time
import os
import shutil
import statistics
import threading
import ctypes
import psutil
import sys

sys.stdout.reconfigure(encoding="utf-8")

COMFY_URL = "http://127.0.0.1:8188"
COMFY_OUTPUT_DIR = r"D:\ComfyUI-aki\ComfyUI-aki-v3\ComfyUI\output"
BENCH_ROOT = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BENCH_ROOT, "images")
RAW_JSON_PATH = os.path.join(BENCH_ROOT, "raw_timings.json")
REPORT_MD_PATH = os.path.join(BENCH_ROOT, "BENCHMARK_REPORT.md")

os.makedirs(IMAGES_DIR, exist_ok=True)


# --- 原生 NVML 显存 + psutil 内存高频峰值监控器 ---
class c_nvmlMemory_t(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


class ResourceMonitor:
    def __init__(self, interval_sec=0.05):
        self.interval = interval_sec
        self.nvml = ctypes.WinDLL("nvml.dll")
        self.nvml.nvmlInit_v2()
        self.handle = ctypes.c_void_p()
        self.nvml.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(self.handle))
        self.comfy_proc = self._find_comfy_process()
        self._running = False
        self._thread = None
        self.peak_vram_mb = 0
        self.peak_proc_ram_gb = 0.0
        self.peak_sys_ram_gb = 0.0

    def _find_comfy_process(self):
        for p in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                cmd = " ".join(p.info.get("cmdline") or []).lower()
                if "main.py" in cmd and ("comfyui" in cmd or "8188" in cmd):
                    return psutil.Process(p.info["pid"])
            except Exception:
                pass
        return None

    def get_current(self):
        mem = c_nvmlMemory_t()
        self.nvml.nvmlDeviceGetMemoryInfo(self.handle, ctypes.byref(mem))
        vram_mb = int(mem.used // (1024 * 1024))
        sys_ram_gb = psutil.virtual_memory().used / (1024 ** 3)
        proc_ram_gb = 0.0
        if self.comfy_proc is None or not self.comfy_proc.is_running():
            self.comfy_proc = self._find_comfy_process()
        if self.comfy_proc:
            try:
                proc_ram_gb = self.comfy_proc.memory_info().rss / (1024 ** 3)
            except Exception:
                pass
        return vram_mb, round(proc_ram_gb, 2), round(sys_ram_gb, 2)

    def _loop(self):
        while self._running:
            vram_mb, proc_gb, sys_gb = self.get_current()
            if vram_mb > self.peak_vram_mb:
                self.peak_vram_mb = vram_mb
            if proc_gb > self.peak_proc_ram_gb:
                self.peak_proc_ram_gb = proc_gb
            if sys_gb > self.peak_sys_ram_gb:
                self.peak_sys_ram_gb = sys_gb
            time.sleep(self.interval)

    def start(self):
        vram_mb, proc_gb, sys_gb = self.get_current()
        self.peak_vram_mb = vram_mb
        self.peak_proc_ram_gb = proc_gb
        self.peak_sys_ram_gb = sys_gb
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        return {
            "vram_mb": self.peak_vram_mb,
            "vram_gb": round(self.peak_vram_mb / 1024.0, 2),
            "proc_ram_gb": round(self.peak_proc_ram_gb, 2),
            "sys_ram_gb": round(self.peak_sys_ram_gb, 2),
        }


MONITOR = ResourceMonitor(interval_sec=0.05)

MODELS = {
    "pure_nvfp4": {
        "file": "qwen-image-2.1-turbo-NVFP4.safetensors",
        "label": "纯 NVFP4 (4.0GB)",
        "short": "pureNVFP4",
    },
    "mixed_nvfp4": {
        "file": "qwen_image_2.1_turbo_nvfp4_mixed.safetensors",
        "label": "混合 NVFP4 (5.4GB)",
        "short": "mixedNVFP4",
    },
}

PRESETS = {
    "1K": "1K 极速档 (1024级 · 日常与文字秒出)",
    "1.3K": "1.3K 官方黄金桶 (1328级 · 通义原生预设)",
    "1.5K": "1.5K 海报排版档 (1536级 · 严格网格对齐)",
    "2K": "2K 原生满血档 (2048级 · 极致细节与微缩字)",
}

RATIOS = {
    "3:4": "3:4 (Portrait Standard)",
    "16:9": "16:9 (Widescreen)",
    "9:16": "9:16 (Portrait Widescreen)",
    "1:1": "1:1 (Square)",
}

PROMPTS = {
    "P1_Exam": {
        "title": "P1: 2026 数学高考试卷",
        "ratio": "3:4",
        "pe_default": False,
        "desc": "密集宋体小字、粗体大标题、数学公式符号",
        "text": "这是一张2026年数学高考试卷的平面设计图，画面直接呈现试卷版面，全出血排版，试卷边缘即为画布边缘。试卷整体为纯白底色，文字和图形均为黑色印刷效果，排版严谨规范，采用标准的高考卷面布局。页面左侧边缘有一条垂直的虚线作为装订密封线。页面顶部居中位置印有黑色粗体大字“绝密★启用前”。其下方是大标题，居中印有“2026年普通高等学校招生全国统一考试 数学”。大标题下方是“注意事项”部分，使用黑色正楷体，分条列出考生须知，内容依次为：“1. 答卷前，考生务必将自己的 姓名、准考证号填写在答题卡上。 2. 回答选择题时，选出每小题答案后， 用铅笔把答题卡上对应题目的答案标号涂黑。 3. 回答非选择题时，将答案写在答题卡上。 写在本试卷上无效。 4. 考试结束后，将本试卷和答题卡一并交回。”注意事项下方是试卷正文部分，采用标准宋体印刷。第一大部分标题为“一、单项选择题：本题共8小题， 每小题5分，共40分。”紧接着是第1题：“1. 已知集合A={x|x²-3x+2=0}， B={x|x²-2x=0}，则A∩B=”，下方列出四个选项：“A. {0} B. {1} C. {2} D. {0, 2}”。第2题为：“2. 若复数z满足z(1+i)=2， 则|z|=”，下方选项为：“A. 1 B. √2 C. 2 D. 2√2”。中间穿插有填空题部分，标题为“二、填空题：本题共4小题， 每小题5分，共20分。”，其中第9题为：“9. 已知向量a=(1, 2)，b=(x, -1)， 若a⊥b，则x=______。”页面下半部分为解答题区域，标题为“三、解答题：本题共5小题，共70分。”第15题内容为：“15.（13分）在△ABC中，内角A, B, C 的对边分别为a, b, c，已知 acosC+ccosA=bsinB。 （1）求角B的大小； （2）若b=2，求△ABC面积的最大值。”题目右侧空白处配有一个简单的黑色线条绘制的三角形几何示意图，顶点分别标有字母“A”、“B”、“C”。整个卷面文字清晰，数学公式排版规范，行距适中，呈现出标准的高考试卷视觉效果。",
    },
    "P2_Castle": {
        "title": "P2: 奥拉维城堡建筑图板",
        "ratio": "16:9",
        "pe_default": False,
        "desc": "精细建筑测绘线稿、多区块排版、右侧信息栏小字",
        "text": "这是一张横向展开的建筑讲解图板，整体以米白羊皮纸质感背景、砂岩色与深棕色线稿为主，画面边缘和各分区使用细深棕色实线框与分割线组织版面。左上角是盾形城堡徽章标志，深棕色线框内有三座塔楼剪影和下方波浪纹，旁边的大标题为“奥拉维城堡”，下方英文副标题为“OLAVINLINNA CASTLE”。标题下方是一段说明文字，以下为完整正文：“奥拉维城堡建于1475年，是芬兰中世纪防御建筑的代表作。城堡位于萨翁林纳湖中的岩岛上，三面环水，地势险要，具有重要的历史与战略价值。”画面主体是一幅大型俯视透视建筑插画，约占页面上半部大部分宽度，表现奥拉维城堡坐落在湖中岩岛上，周围是淡灰蓝色水面和岩石岸线，城堡由多座圆形石塔、厚重环墙、内庭院、桥梁和入口组成，线稿细密，石材纹理、窗洞、箭孔和屋顶铜板分缝清晰可见。主视觉上有深棕色圆形编号点与虚线引导标注，桥梁入口处标注“1 主入口”，左侧内院标注“2 第一庭院”，中央内院标注“2 第二庭院”，中央高大的圆形主塔上方标注“4 圣奥拉夫塔（主塔）”，右侧圆塔标注“5 小塔楼”，右侧外墙区域标注“6 环墙”。主视觉右上方还有一个简洁罗盘，圆形细线外框内是深棕色北向指针，上方文字为“N”。右侧竖向信息栏由多个矩形框组成，顶部框标题为“基本信息”，内部每行左侧配有线性小图标，依次是钟表、城堡、建筑山墙、立方体材料图标和面积立方体图标，对应文字为“建造时间：1475年”、“建筑类型：水上城堡”、“建筑风格：中世纪防御建筑”、“建筑材料：当地花岗岩”、“建筑面积：约1,780㎡”。其下方框标题为“平面概览”，包含一幅小型平面图，使用深棕线条描绘三个圆塔、连墙、入口桥与内部建筑轮廓，下方有比例尺，刻度文字为“0”、“10”、“20”、“30m”，右下角还有一个小圆形方向符号。再下方框标题为“立面比例分析”，展示一幅小型立面比例示意，虚线水平标尺旁标有“1.0”、“2.2”、“1.0”，用以比较塔顶、塔身和基座等高度关系。页面中部偏下是一幅大型南立面建筑线稿，标题为“南立面图 1:200”，立面显示三座圆塔与连接墙体、入口桥、岩石基座和细小人群尺度，塔顶为锥形铜色屋面，塔身为浅砂岩色块石墙面。立面上方有水平尺寸标注“18.2m”、“24.6m”、“18.2m”，左侧有竖向高度标注“32.8m”、“21.6m（塔身）”、“11.2m（墙体）”。立面右侧用虚线引出编号说明，依次为“1 塔顶 铜质尖顶，便于排水”、“2 射击孔 圆形孔洞，用于防御”、“3 拱形窗 竖向狭长窗，节奏均匀”、“4 墙体开窗节奏 小窗与箭孔交替布置，形成稳定的律”、“5 墙体材料 当地花岗岩砌筑，厚实坚固，适应水上环境”、“6 基座 依岩而建，直接与基岩结合，增强稳定性”。右侧中下部是“局部放大详图”区域，采用三个横向分隔的细框卡片，每个卡片左侧是放大的构造线稿，右侧是引线文字。第一项标题为“1 塔顶构造 1:20”，图中放大圆塔顶部，标注“铜板屋面”、“木构基层”、“砖石檐口”。第二项标题为“2 拱形窗构造 1:20”，展示拱形窗节点，标注“花岗岩拱券”、“花岗岩侧壁”、“木质窗框”、“铁制格栅”。第三项标题为“3 墙体与箭孔构造 1:20”，展示块石墙体与狭长箭孔，标注“花岗岩块石”、“箭孔”、“内侧抹灰”。页面底部左侧是“入口人流分析”模块，左边为图例，深棕实线箭头代表“主要人流动线”，深棕虚线箭头代表“次要人流动线”，深棕圆点代表“停留节点”，实心箭头代表“人流方向”；右侧小插图以俯视方式表现桥梁、入口门楼、岛屿平台和众多微型人物剪影，箭头沿桥面与门楼前广场组织人流方向。底部中间是“材料图例”模块，使用五个圆形材质样本展示石材、砖、铜板、木材和铁件的纹理，文字分别为“花岗岩 主要墙体材料”、“砖砌 拱券及局部构造”、“铜板 屋面覆盖材料”、“木材 门窗及结构构件”、“铁件 门窗格栅及构件”。底部右侧是“构造剖面示意 1:100”，展示塔楼剖面与连接墙体，塔顶木屋架、楼板、塔身空间、墙体厚度和岩石基础以剖切方式呈现，右侧编号引线标注为“1 塔顶结构”、“2 木质屋架”、“3 塔身空间”、“4 楼板结构”、“5 墙体厚度（3.5~4.5m）”、“6 基岩基础”。整体图面像专业建筑展示板，线条清晰、图例完整、编号系统统一，带有历史建筑手绘测绘图与信息可视化结合的风格。",
    },
    "P3_AppUI": {
        "title": "P3: Wanderlust 旅行 App 界面",
        "ratio": "9:16",
        "pe_default": False,
        "desc": "手机竖屏 UI 界面、图标、圆角卡片与英文小字",
        "text": 'The image is a vertical smartphone screenshot of a modern travel planner app interface named "Wanderlust", designed with a clean, light-themed aesthetic, soft shadows, and rounded corners, fully occupying the screen. The top edge features a standard status bar with black text "9:41" on the left, and on the right, black cellular signal bars, a Wi-Fi icon, and a battery icon showing "85%". Below the status bar is a top navigation bar. On the left, a circular user avatar shows a smiling young woman with a light blue background. Next to it, bold black text reads "Good morning, Sarah". On the far right, a dark gray rounded square icon contains a white gear symbol for settings.\n\nThe main hero section is a large, rounded-corner card with a soft drop shadow. The background is a vibrant, high-quality photograph of the Santorini coastline with white cubic buildings and bright blue domes against a clear sky. Overlaid on the bottom left of the image is a semi-transparent dark gradient. Text inside reads "Santorini, Greece" in large bold white font, followed by "Oct 12 - Oct 19" in smaller white font. On the bottom right, a solid white rounded button contains black text "View Itinerary". At the very top right of this card, a small white pill-shaped badge contains black text "5 Days Left".\n\nBelow the hero card is a section titled "Plan Your Trip" in bold dark gray text. Underneath is a row of four evenly spaced, light gray rounded square cards with subtle borders. The first card contains a blue airplane icon and black text "Flights" below. The second card contains an orange bed icon and black text "Hotels" below. The third card contains a green camera icon and black text "Activities" below. The fourth card contains a purple train icon and black text "Transfers" below.\n\nThe next section is titled "Suggested for You" in bold black text, with a blue text link "See All" aligned to the right. Below is a vertical list of three destination recommendation cards, each with a white background and a soft shadow. The first card has a square thumbnail of the Eiffel Tower on the left. The right side has the title "Paris" in bold black, the subtitle "Romantic Getaway" in dark gray, and a bottom row containing a yellow star icon, black text "4.8", and a small light gray pill with black text "Flights from $450". The second card has a square thumbnail of the Colosseum on the left. The right side has the title "Rome" in bold black, the subtitle "Historical Tour" in dark gray, and a bottom row containing a yellow star icon, black text "4.7", and a small light gray pill with black text "Flights from $520". The third card has a square thumbnail of a Swiss Alpine village on the left. The right side has the title "Zermatt" in bold black, the subtitle "Mountain Retreat" in dark gray, and a bottom row containing a yellow star icon, black text "4.9", and a small light gray pill with black text "Flights from $600".\n\nBelow the suggestions is a section titled "Past Trips" in bold black text. This section contains a vertical list of two past trip history cards. The first card features a rectangular thumbnail of the neon-lit Shibuya crossing on the left. The middle area displays the title "Tokyo, Japan" in bold black text and the dates "Aug 05 - Aug 12" in dark gray text. The right side features a solid blue rounded button with white text "Rebook". The second card features a rectangular thumbnail of the red rocks of the Grand Canyon on the left. The middle area displays the title "Arizona, USA" in bold black text and the dates "May 10 - May 15" in dark gray text. The right side features a solid blue rounded button with white text "Rebook".\n\nThe bottom layer is a fixed bottom navigation bar with a pure white background and a subtle top shadow, spanning the full width. It contains four evenly spaced interactive elements. The first element is a solid blue home icon with blue text "Home" directly below. The second element is a gray magnifying glass icon with gray text "Explore" directly below. The third element is a gray calendar icon with gray text "Bookings" directly below. The fourth element is a gray person icon with gray text "Profile" directly below. The entire interface uses a consistent design language with uniform corner radii, a harmonious color palette of white, light gray, dark gray, and vibrant blue accents, and clear typographic hierarchy ensuring all text is perfectly readable.',
    },
    "P4_Bride": {
        "title": "P4: 动漫婚纱角色立绘",
        "ratio": "3:4",
        "pe_default": False,
        "desc": "无字纯画质：二次元五官、多层薄纱材质与柔和光影",
        "text": "这是一张带有透明度的RGBA格式图像，呈现一位身着华丽婚纱的动漫风格女性角色。画面中无任何可识别的文字信息。角色为年轻女性，拥有白皙肌肤、红润脸颊与明亮红色眼眸，面带温柔微笑，表情甜美而幸福。她头戴精致银色皇冠与白色蕾丝头纱，金色长发自然垂落肩头，发丝柔顺且富有光泽。身穿多层荷叶边设计的粉白渐变抹胸婚纱，裙摆蓬松飘逸，层次丰富，材质轻盈如花瓣般展开；胸前手持一束由白色玫瑰与绿色叶片组成的捧花，花束饱满，绿叶点缀其间，增添清新感。双腿修长，穿着透明薄纱质感的过膝袜与饰有玫瑰图案的白色高跟鞋，姿态优雅站立，双手轻握捧花置于腹前。整体构图为全身正面视角，色彩柔和梦幻，以粉、白、绿为主色调，光影细腻，具有典型的日系二次元插画美学风格，适用于游戏立绘、视觉小说或数字艺术收藏等场景。",
    },
    "P5_Temple": {
        "title": "P5: 祈年殿拆解科普卡 (短提示词)",
        "ratio": "3:4",
        "pe_default": True,
        "desc": "中式书法大字 + 短提示词扩写补全建筑构件名称",
        "text": '竖版祈年殿拆解科普卡，纯白色宣纸底色，左竖排书法"祈年殿"、小字天坛与红印章，附线稿和简介；主体从上到下分层拆解全建筑构件，右侧虚线序号标注名称，中式手绘，配色棕、深蓝、朱红、米白。',
    },
    "P6_Monster": {
        "title": "P6: 巨兽对决 3DCG 故事板 (短提示词)",
        "ratio": "1:1",
        "pe_default": True,
        "desc": "1:1 方图多格分镜、3DCG 光影特效、短提示词扩写",
        "text": "核心大纲 1. 场景：3DCG质感，废弃都市遗迹（黄昏），尘土漫天，绯红色夕阳，地面有巨大脚印。 2. 人物：炎狱巨蜥（熔岩鳞片、喷烈焰）、霜牙巨象（雪白、喷寒气）、幸存人类（远景蜷缩）。 3. 剧情脉络： （1）对峙（0-20秒）：两只巨兽在废墟对峙，人类蜷缩避险，氛围压迫。 （2）开战（21-50秒）：炎狱巨蜥喷烈焰，霜牙巨象喷寒气对抗，引发冲击，高楼坍塌。 （3）激战（51-90秒）：双方互相攻击，均受重伤，嘶吼震彻废墟。 （4）落幕（91-110秒）：两只巨兽重伤对峙，镜头拉远，定格巨兽与废墟，留下悬念",
    },
}


def build_prompt_graph(unet_file, preset_key, ratio_key, steps, prompt_text, seed, pe=False, fbcache=False, prefix="bench"):
    return {
        "13": {
            "class_type": "ResolutionSelector",
            "inputs": {
                "aspect_ratio": RATIOS[ratio_key],
                "qwen_preset": PRESETS[preset_key],
            },
        },
        "459:451": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": unet_file,
                "weight_dtype": "default",
            },
        },
        "459:453": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": "qwen3vl_8b_w4a8_lmhead4bit.safetensors",
                "type": "qwen_image",
                "device": "default",
            },
        },
        "459:454": {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": "qwen_image_2.1_vae_bf16.safetensors",
            },
        },
        "459:456": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": ["13", 0],
                "height": ["13", 1],
                "batch_size": 1,
            },
        },
        "459:457": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["459:458", 0],
                "vae": ["459:454", 0],
            },
        },
        "459:458": {
            "class_type": "SamplerCustom",
            "inputs": {
                "add_noise": True,
                "noise_seed": int(seed),
                "cfg": 1.0,
                "model": ["459:469", 0],
                "positive": ["459:474", 0],
                "negative": ["459:474", 1],
                "sampler": ["459:482", 0],
                "sigmas": ["459:485", 0],
                "latent_image": ["459:468", 0],
            },
        },
        "459:468": {
            "class_type": "ComfySwitchNode",
            "inputs": {
                "switch": True,
                "on_false": ["459:474", 2],
                "on_true": ["459:456", 0],
            },
        },
        "459:469": {
            "class_type": "QwenImage21Cache",
            "inputs": {
                "device": "auto",
                "dtype": "int8",
                "enable_fbcache": bool(fbcache),
                "model": ["459:486", 0],
            },
        },
        "459:474": {
            "class_type": "TextEncodeQwenImage21",
            "inputs": {
                "prompt": prompt_text,
                "negative_prompt": "",
                "resolution": 0,
                "enable_prompt_enhance": bool(pe),
                "clip": ["459:453", 0],
                "vae": ["459:454", 0],
            },
        },
        "459:481": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "lora_name": "qwen_image_2.1_turbo_lora_avg_rank_178_bf16.safetensors",
                "strength_model": 0.0,
                "model": ["459:451", 0],
            },
        },
        "459:482": {
            "class_type": "KSamplerSelect",
            "inputs": {"sampler_name": "euler"},
        },
        "459:483": {
            "class_type": "ManualSigmas",
            "inputs": {
                "sigmas": "1.0, 0.972236, 0.937186, 0.89508, 0.805499, 0.514892, 0.0",
                "steps": int(steps),
            },
        },
        "459:484": {
            "class_type": "BasicScheduler",
            "inputs": {
                "scheduler": "simple",
                "steps": int(steps),
                "denoise": 1.0,
                "model": ["459:469", 0],
            },
        },
        "459:485": {
            "class_type": "ComfySwitchNode",
            "inputs": {
                "switch": ["13", 2],
                "on_false": ["459:500", 0],
                "on_true": ["459:505", 0],
            },
        },
        "459:486": {
            "class_type": "ModelAttentionBackend",
            "inputs": {
                "attention": "comfy kitchen attention",
                "model": ["459:481", 0],
            },
        },
        "459:490": {
            "class_type": "ComfyMathExpression",
            "inputs": {"expression": "a <= 6", "values.a": int(steps)},
        },
        "459:491": {
            "class_type": "ComfyMathExpression",
            "inputs": {
                "expression": "a * b > 3000000",
                "values.a": ["13", 0],
                "values.b": ["13", 1],
            },
        },
        "459:492": {
            "class_type": "ComfyMathExpression",
            "inputs": {
                "expression": "(a % 128 == 0) and (b % 128 == 0)",
                "values.a": ["13", 0],
                "values.b": ["13", 1],
            },
        },
        "459:493": {
            "class_type": "ManualSigmas",
            "inputs": {
                "sigmas": "1.0, 0.934262, 0.848546, 0.741433, 0.584844, 0.328448, 0.0",
                "steps": int(steps),
            },
        },
        "459:494": {
            "class_type": "ManualSigmas",
            "inputs": {
                "sigmas": "1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0.0",
                "steps": int(steps),
            },
        },
        "459:495": {
            "class_type": "ManualSigmas",
            "inputs": {
                "sigmas": "1.0, 0.972236, 0.937186, 0.89508, 0.805499, 0.514892, 0.0",
                "steps": int(steps),
            },
        },
        "459:496": {
            "class_type": "ManualSigmas",
            "inputs": {
                "sigmas": "1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0.0",
                "steps": int(steps),
            },
        },
        "459:497": {
            "class_type": "ManualSigmas",
            "inputs": {
                "sigmas": "1.0, 0.972236, 0.937186, 0.89508, 0.805499, 0.514892, 0.0",
                "steps": int(steps),
            },
        },
        "459:498": {
            "class_type": "ManualSigmas",
            "inputs": {
                "sigmas": "1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0.0",
                "steps": int(steps),
            },
        },
        "459:500": {
            "class_type": "ComfySwitchNode",
            "inputs": {
                "switch": ["459:490", 2],
                "on_false": ["459:484", 0],
                "on_true": ["459:493", 0],
            },
        },
        "459:501": {
            "class_type": "ComfySwitchNode",
            "inputs": {
                "switch": ["459:490", 2],
                "on_false": ["459:494", 0],
                "on_true": ["459:495", 0],
            },
        },
        "459:502": {
            "class_type": "ComfySwitchNode",
            "inputs": {
                "switch": ["459:490", 2],
                "on_false": ["459:496", 0],
                "on_true": ["459:497", 0],
            },
        },
        "459:503": {
            "class_type": "ComfySwitchNode",
            "inputs": {
                "switch": ["459:490", 2],
                "on_false": ["459:498", 0],
                "on_true": ["459:483", 0],
            },
        },
        "459:504": {
            "class_type": "ComfySwitchNode",
            "inputs": {
                "switch": ["459:492", 2],
                "on_false": ["459:501", 0],
                "on_true": ["459:502", 0],
            },
        },
        "459:505": {
            "class_type": "ComfySwitchNode",
            "inputs": {
                "switch": ["459:491", 2],
                "on_false": ["459:504", 0],
                "on_true": ["459:503", 0],
            },
        },
        "461": {
            "class_type": "SaveAnimatedWEBP",
            "inputs": {
                "filename_prefix": f"benchmark_20261013/{prefix}",
                "fps": 6.0,
                "lossless": False,
                "quality": 95,
                "method": "fastest",
                "images": ["459:457", 0],
            },
        },
    }


def run_single_job(graph, target_image_name=None):
    payload = json.dumps({"prompt": graph}).encode("utf-8")
    req = urllib.request.Request(
        f"{COMFY_URL}/prompt",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    MONITOR.start()
    t0 = time.perf_counter()
    resp = json.loads(urllib.request.urlopen(req, timeout=30).read())
    prompt_id = resp["prompt_id"]

    while True:
        time.sleep(0.15)
        hist_raw = urllib.request.urlopen(f"{COMFY_URL}/history/{prompt_id}", timeout=30).read()
        hist = json.loads(hist_raw)
        if prompt_id in hist and hist[prompt_id].get("status", {}).get("completed", False):
            entry = hist[prompt_id]
            break

    wall_elapsed = time.perf_counter() - t0
    peaks = MONITOR.stop()

    msgs = entry.get("status", {}).get("messages", [])
    ts_start = None
    ts_end = None
    for m_type, m_data in msgs:
        if m_type == "execution_start":
            ts_start = m_data.get("timestamp")
        elif m_type == "execution_success":
            ts_end = m_data.get("timestamp")

    if ts_start and ts_end:
        exact_sec = (ts_end - ts_start) / 1000.0
    else:
        exact_sec = wall_elapsed

    out_imgs = entry.get("outputs", {}).get("461", {}).get("images", [])
    saved_rel_path = None
    if out_imgs:
        img_info = out_imgs[0]
        src_path = os.path.join(COMFY_OUTPUT_DIR, img_info.get("subfolder", ""), img_info["filename"])
        if target_image_name and os.path.exists(src_path):
            dst_path = os.path.join(IMAGES_DIR, target_image_name)
            shutil.copy2(src_path, dst_path)
            saved_rel_path = f"images/{target_image_name}"

    return round(exact_sec, 3), saved_rel_path, peaks


def load_state():
    if os.path.exists(RAW_JSON_PATH):
        try:
            return json.load(open(RAW_JSON_PATH, "r", encoding="utf-8"))
        except Exception:
            pass
    return {"part1": {}, "part2": {}, "part3": {}, "cold_start": {}}


def save_state(state):
    with open(RAW_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    generate_markdown_report(state)


def generate_markdown_report(state):
    lines = []
    lines.append("# Qwen-Image 2.1 Turbo 全面基准测试报告 (RTX 5060 Laptop 8GB)")
    lines.append("")
    lines.append("- **测试显卡**：NVIDIA GeForce RTX 5060 Laptop GPU (8151 MB VRAM)")
    lines.append("- **软件环境**：Windows 11 / Python 3.13.11 / PyTorch 2.9.1+cu130 / ComfyUI v0.37.1")
    lines.append("- **注意力与前缀缓存**：`comfy kitchen attention` (INT8) + `QwenImage21Cache` (`auto`, `int8`)")
    lines.append("- **测试方法**：每次切换模型或切换分辨率档位先执行 **1 次预热 (Warm-up)**；稳态测速采用 **`seed = 42, 43, 44` 连续运行 3 次取平均值 (Mean) ± 标准差 (Std)**；通过 `nvml.dll` 与 `psutil` 每 `50ms` 采样记录 **GPU 显存峰值 (`VRAM Peak`)** 与 **ComfyUI 进程内存峰值 (`Proc RAM Peak`)**；画质对比图统一固定 **`seed = 42`**。")
    lines.append("")

    if state.get("cold_start"):
        lines.append("## 一、模型切换与冷启动耗时 (Cold Start)")
        lines.append("")
        lines.append("| 扩散模型 | 权重体积 | 首次加载/切换预热耗时 | 显存峰值 (VRAM Peak) | ComfyUI 内存峰值 (Proc RAM) | 系统总内存峰值 (Sys RAM) |")
        lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
        for m_key, val in state["cold_start"].items():
            m_label = MODELS[m_key]["label"]
            size_str = "4.00 GB" if m_key == "pure_nvfp4" else "5.43 GB"
            if isinstance(val, dict):
                lines.append(
                    f"| `{MODELS[m_key]['file']}` ({m_label}) | {size_str} | **{val['sec']:.2f}s** | {val['peaks']['vram_mb']} MB ({val['peaks']['vram_gb']:.2f} GB) | {val['peaks']['proc_ram_gb']:.2f} GB | {val['peaks']['sys_ram_gb']:.2f} GB |"
                )
            else:
                lines.append(f"| `{MODELS[m_key]['file']}` ({m_label}) | {size_str} | **{val:.2f}s** | - | - | - |")
        lines.append("")

    lines.append("## 二、Part 1：模型精度 × 全分辨率档位 × 步数 性能与显存/内存基准表 (P1 数学高考试卷 `3:4`)")
    lines.append("")
    lines.append("| 分辨率 (`3:4`) | 步数 | 混合 NVFP4 (均值 ± 标准差) | 纯 NVFP4 (均值 ± 标准差) | 纯 NVFP4 提速 | 混合 NVFP4 显存/内存峰值 | 纯 NVFP4 显存/内存峰值 | 混合 NVFP4 样张 | 纯 NVFP4 样张 |")
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

    p1 = state.get("part1", {})
    for res_key in ["1K", "1.3K", "1.5K", "2K"]:
        for steps in [4, 6, 8]:
            k_mix = f"mixed_nvfp4_{res_key}_{steps}step"
            k_pure = f"pure_nvfp4_{res_key}_{steps}step"
            d_mix = p1.get(k_mix)
            d_pure = p1.get(k_pure)

            s_mix = f"**{d_mix['mean']:.2f}s** ± {d_mix['std']:.2f}s" if d_mix else "待测..."
            s_pure = f"**{d_pure['mean']:.2f}s** ± {d_pure['std']:.2f}s" if d_pure else "待测..."

            if d_mix and d_pure and d_mix["mean"] > 0:
                diff = d_mix["mean"] - d_pure["mean"]
                pct = (diff / d_mix["mean"]) * 100
                speedup = f"**-{diff:.2f}s (-{pct:.1f}%)**"
            else:
                speedup = "-"

            m_mix = f"VRAM: {d_mix['peak_vram_mb']}MB / RAM: {d_mix['peak_proc_ram_gb']:.2f}G" if d_mix else "-"
            m_pure = f"VRAM: {d_pure['peak_vram_mb']}MB / RAM: {d_pure['peak_proc_ram_gb']:.2f}G" if d_pure else "-"

            img_mix = f"[样张]({d_mix['image']})" if (d_mix and d_mix.get("image")) else "-"
            img_pure = f"[样张]({d_pure['image']})" if (d_pure and d_pure.get("image")) else "-"
            lines.append(f"| **{res_key}** | `{steps}` 步 | {s_mix} | {s_pure} | {speedup} | {m_mix} | {m_pure} | {img_mix} | {img_pure} |")
    lines.append("")

    lines.append("## 三、Part 2：全提示词 × 专属常用比例 (1K / 2K × 4步 / 6步) 对比表 (`seed=42`)")
    lines.append("")
    lines.append("| 提示词场景 | 比例 | 分辨率 | 步数 | 混合 NVFP4 (耗时 / 显存 / 内存) | 纯 NVFP4 (耗时 / 显存 / 内存) | 混合 NVFP4 样张 | 纯 NVFP4 样张 |")
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

    p2 = state.get("part2", {})
    for p_key in ["P1_Exam", "P2_Castle", "P3_AppUI", "P4_Bride", "P5_Temple", "P6_Monster"]:
        p_info = PROMPTS[p_key]
        for res_key in ["1K", "2K"]:
            for steps in [4, 6]:
                if p_key == "P1_Exam":
                    d_mix = p1.get(f"mixed_nvfp4_{res_key}_{steps}step")
                    d_pure = p1.get(f"pure_nvfp4_{res_key}_{steps}step")
                    t_mix = f"**{d_mix['mean']:.2f}s** ({d_mix['peak_vram_mb']}MB / {d_mix['peak_proc_ram_gb']:.2f}G)" if d_mix else "待测..."
                    t_pure = f"**{d_pure['mean']:.2f}s** ({d_pure['peak_vram_mb']}MB / {d_pure['peak_proc_ram_gb']:.2f}G)" if d_pure else "待测..."
                    img_mix = f"[样张]({d_mix['image']})" if (d_mix and d_mix.get("image")) else "-"
                    img_pure = f"[样张]({d_pure['image']})" if (d_pure and d_pure.get("image")) else "-"
                else:
                    d_mix = p2.get(f"mixed_nvfp4_{p_key}_{res_key}_{steps}step")
                    d_pure = p2.get(f"pure_nvfp4_{p_key}_{res_key}_{steps}step")
                    t_mix = f"**{d_mix['sec']:.2f}s** ({d_mix['peaks']['vram_mb']}MB / {d_mix['peaks']['proc_ram_gb']:.2f}G)" if d_mix else "待测..."
                    t_pure = f"**{d_pure['sec']:.2f}s** ({d_pure['peaks']['vram_mb']}MB / {d_pure['peaks']['proc_ram_gb']:.2f}G)" if d_pure else "待测..."
                    img_mix = f"[样张]({d_mix['image']})" if (d_mix and d_mix.get("image")) else "-"
                    img_pure = f"[样张]({d_pure['image']})" if (d_pure and d_pure.get("image")) else "-"
                lines.append(f"| **{p_info['title']}** | `{p_info['ratio']}` | **{res_key}** | `{steps}` 步 | {t_mix} | {t_pure} | {img_mix} | {img_pure} |")
    lines.append("")

    lines.append("## 四、Part 3：外层开关 (`prompt_enhance` 扩写 & `fbcache` 缓存) 专项测试")
    lines.append("")
    lines.append("| 测试专项 | 提示词与比例 | 分辨率与步数 | 开关状态 | 实测耗时 (s) | 显存峰值 (VRAM) | 进程内存峰值 (RAM) | 样张链接 |")
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    p3 = state.get("part3", {})
    for k, item in p3.items():
        img_link = f"[查看样张]({item['image']})" if item.get("image") else "-"
        pk = item.get("peaks", {})
        vram_s = f"{pk.get('vram_mb', '-')} MB"
        ram_s = f"{pk.get('proc_ram_gb', '-')} GB"
        lines.append(f"| {item['category']} | {item['prompt_title']} (`{item['ratio']}`) | {item['res']} / `{item['steps']}`步 | `{item['toggle']}` | **{item['sec']:.2f}s** | {vram_s} | {ram_s} | {img_link} |")
    lines.append("")

    with open(REPORT_MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    state = load_state()
    start_all = time.time()

    for m_key in ["pure_nvfp4", "mixed_nvfp4"]:
        m_info = MODELS[m_key]
        print(f"\n==========================================")
        print(f"开始测试模型: {m_info['label']} ({m_info['file']})")
        print(f"==========================================")

        # 1. 模型切换/冷启动预热
        if m_key not in state["cold_start"]:
            print(f"[Cold Start] 正在执行模型预热 ({m_info['short']})...")
            g_cold = build_prompt_graph(
                m_info["file"], "1K", "3:4", 4, PROMPTS["P1_Exam"]["text"], seed=100, pe=False, fbcache=False, prefix=f"warmup_{m_info['short']}"
            )
            sec_cold, _, peaks_cold = run_single_job(g_cold, target_image_name=None)
            state["cold_start"][m_key] = {"sec": sec_cold, "peaks": peaks_cold}
            save_state(state)
            print(f"  -> 冷启动/首次预热完成: {sec_cold:.2f}s | VRAM Peak: {peaks_cold['vram_mb']}MB | Proc RAM: {peaks_cold['proc_ram_gb']}GB")

        # 2. Part 1: P1_Exam 全分辨率 (1K, 1.3K, 1.5K, 2K) x 全步数 (4, 6, 8) + 预热 + 3次均值
        for res_key in ["1K", "1.3K", "1.5K", "2K"]:
            res_warmed = False
            for steps in [4, 6, 8]:
                rec_key = f"{m_key}_{res_key}_{steps}step"
                if rec_key in state["part1"]:
                    print(f"[Skip] 已存在记录: {rec_key}")
                    res_warmed = True
                    continue

                warmup_sec = 0.0
                if not res_warmed:
                    print(f"[Warm-up] {m_info['short']} | {res_key} (3:4) 档位首次预热...")
                    g_warm = build_prompt_graph(
                        m_info["file"], res_key, "3:4", steps, PROMPTS["P1_Exam"]["text"], seed=200 + steps, pe=False, fbcache=False, prefix=f"warmup_{m_info['short']}_{res_key}"
                    )
                    warmup_sec, _, pk_warm = run_single_job(g_warm, target_image_name=None)
                    res_warmed = True
                    print(f"  -> 档位预热耗时: {warmup_sec:.2f}s | VRAM Peak: {pk_warm['vram_mb']}MB")

                runs = []
                vram_peaks = []
                proc_ram_peaks = []
                sys_ram_peaks = []
                saved_img = None
                for idx, seed in enumerate([42, 43, 44]):
                    img_name = f"Part1_P1_Exam_3x4_{res_key}_{steps}step_{m_info['short']}.webp" if seed == 42 else None
                    g_run = build_prompt_graph(
                        m_info["file"], res_key, "3:4", steps, PROMPTS["P1_Exam"]["text"], seed=seed, pe=False, fbcache=False,
                        prefix=f"P1_{res_key}_{steps}step_{m_info['short']}_s{seed}"
                    )
                    sec, rel_p, pk = run_single_job(g_run, target_image_name=img_name)
                    runs.append(sec)
                    vram_peaks.append(pk["vram_mb"])
                    proc_ram_peaks.append(pk["proc_ram_gb"])
                    sys_ram_peaks.append(pk["sys_ram_gb"])
                    if rel_p:
                        saved_img = rel_p
                    print(f"  [Part1] {m_info['short']} | {res_key} | {steps}步 | Run {idx+1}/3 (seed={seed}): {sec:.2f}s | VRAM: {pk['vram_mb']}MB | RAM: {pk['proc_ram_gb']}GB")

                mean_val = round(statistics.mean(runs), 3)
                std_val = round(statistics.stdev(runs), 3) if len(runs) > 1 else 0.0
                state["part1"][rec_key] = {
                    "model": m_key,
                    "res": res_key,
                    "steps": steps,
                    "warmup": warmup_sec,
                    "runs": runs,
                    "mean": mean_val,
                    "std": std_val,
                    "peak_vram_mb": max(vram_peaks),
                    "peak_proc_ram_gb": max(proc_ram_peaks),
                    "peak_sys_ram_gb": max(sys_ram_peaks),
                    "image": saved_img,
                }
                save_state(state)
                print(f"  => 汇总 {rec_key}: Mean={mean_val:.2f}s ± {std_val:.2f}s | Peak VRAM={max(vram_peaks)}MB | Peak Proc RAM={max(proc_ram_peaks):.2f}GB")

        # 3. Part 2: 其余 5 条提示词 (P2~P6) 按各自专属比例跑 1K / 2K x 4步 / 6步 (seed=42)
        for p_key in ["P2_Castle", "P3_AppUI", "P4_Bride", "P5_Temple", "P6_Monster"]:
            p_info = PROMPTS[p_key]
            ratio_key = p_info["ratio"]
            ratio_tag = ratio_key.replace(":", "x")
            for res_key in ["1K", "2K"]:
                for steps in [4, 6]:
                    rec_key = f"{m_key}_{p_key}_{res_key}_{steps}step"
                    if rec_key in state["part2"]:
                        print(f"[Skip] 已存在记录: {rec_key}")
                        continue

                    img_name = f"Part2_{p_key}_{ratio_tag}_{res_key}_{steps}step_{m_info['short']}.webp"
                    g_run = build_prompt_graph(
                        m_info["file"], res_key, ratio_key, steps, p_info["text"], seed=42, pe=False, fbcache=False,
                        prefix=f"Part2_{p_key}_{res_key}_{steps}step_{m_info['short']}"
                    )
                    sec, rel_p, pk = run_single_job(g_run, target_image_name=img_name)
                    state["part2"][rec_key] = {
                        "model": m_key,
                        "prompt": p_key,
                        "ratio": ratio_key,
                        "res": res_key,
                        "steps": steps,
                        "sec": sec,
                        "peaks": pk,
                        "image": rel_p,
                    }
                    save_state(state)
                    print(f"  [Part2] {m_info['short']} | {p_key} ({ratio_key}) | {res_key} | {steps}步: {sec:.2f}s | VRAM: {pk['vram_mb']}MB | RAM: {pk['proc_ram_gb']}GB")

        # 4. Part 3: 在 pure_nvfp4 下测外层开关 (prompt_enhance 开 vs 关, fbcache 开 vs 关)
        if m_key == "pure_nvfp4":
            for p_key in ["P5_Temple", "P6_Monster"]:
                p_info = PROMPTS[p_key]
                ratio_key = p_info["ratio"]
                ratio_tag = ratio_key.replace(":", "x")
                for pe_flag in [False, True]:
                    pe_str = "PE_On" if pe_flag else "PE_Off"
                    rec_key = f"PE_{p_key}_1.3K_6step_{pe_str}"
                    if rec_key in state["part3"]:
                        continue
                    img_name = f"Part3_{p_key}_{ratio_tag}_1.3K_6step_{pe_str}_pureNVFP4.webp"
                    g_run = build_prompt_graph(
                        m_info["file"], "1.3K", ratio_key, 6, p_info["text"], seed=42, pe=pe_flag, fbcache=False,
                        prefix=f"Part3_{rec_key}"
                    )
                    sec, rel_p, pk = run_single_job(g_run, target_image_name=img_name)
                    state["part3"][rec_key] = {
                        "category": "Prompt Enhance 扩写对比",
                        "prompt_title": p_info["title"],
                        "ratio": ratio_key,
                        "res": "1.3K",
                        "steps": 6,
                        "toggle": f"prompt_enhance={pe_flag}",
                        "sec": sec,
                        "peaks": pk,
                        "image": rel_p,
                    }
                    save_state(state)
                    print(f"  [Part3-PE] {p_key} | 1.3K 6步 | {pe_str}: {sec:.2f}s | VRAM: {pk['vram_mb']}MB | RAM: {pk['proc_ram_gb']}GB")

            for p_key in ["P1_Exam", "P2_Castle"]:
                p_info = PROMPTS[p_key]
                ratio_key = p_info["ratio"]
                ratio_tag = ratio_key.replace(":", "x")
                for fb_flag in [False, True]:
                    fb_str = "FBCache_On" if fb_flag else "FBCache_Off"
                    rec_key = f"FBC_{p_key}_2K_6step_{fb_str}"
                    if rec_key in state["part3"]:
                        continue
                    img_name = f"Part3_{p_key}_{ratio_tag}_2K_6step_{fb_str}_pureNVFP4.webp"
                    g_run = build_prompt_graph(
                        m_info["file"], "2K", ratio_key, 6, p_info["text"], seed=42, pe=False, fbcache=fb_flag,
                        prefix=f"Part3_{rec_key}"
                    )
                    sec, rel_p, pk = run_single_job(g_run, target_image_name=img_name)
                    state["part3"][rec_key] = {
                        "category": "FBCache 一阶特征缓存对比",
                        "prompt_title": p_info["title"],
                        "ratio": ratio_key,
                        "res": "2K",
                        "steps": 6,
                        "toggle": f"fbcache={fb_flag}",
                        "sec": sec,
                        "peaks": pk,
                        "image": rel_p,
                    }
                    save_state(state)
                    print(f"  [Part3-FBCache] {p_key} | 2K 6步 | {fb_str}: {sec:.2f}s | VRAM: {pk['vram_mb']}MB | RAM: {pk['proc_ram_gb']}GB")

    total_min = (time.time() - start_all) / 60.0
    print(f"\n==========================================")
    print(f"全部基准测试完成！总耗时: {total_min:.1f} 分钟")
    print(f"报告已生成至: {REPORT_MD_PATH}")
    print(f"样张目录: {IMAGES_DIR}")
    print(f"==========================================")


if __name__ == "__main__":
    main()
