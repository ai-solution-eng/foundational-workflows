all_inputs = [
    {"text": "A woman playing with her dog on a beach at sunset."},
    {"text": "Pet owner training dog outdoors near water."},
    {"text": "Woman surfing on waves during a sunny day."},
    {"text": "City skyline view from a high-rise building at night."},
    {
        "text": (
            "A woman shares a joyful moment with her golden retriever on a"
            " sun-drenched beach at sunset, as the dog offers its paw in a"
            " heartwarming display of companionship and trust."
        )
    },
    {"image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"},
    {
        "text": (
            "A woman shares a joyful moment with her golden retriever on a"
            " sun-drenched beach at sunset, as the dog offers its paw in a"
            " heartwarming display of companionship and trust."
        ),
        "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
    },
]

queries = all_inputs[:4]
documents = all_inputs[4:]
