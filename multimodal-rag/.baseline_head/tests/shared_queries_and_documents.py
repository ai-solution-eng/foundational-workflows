from pathlib import Path

all_inputs = [
    {"text": "The aurora borealis over a snowy mountain"},
    {"image": "https://fastly.picsum.photos/id/901/3517/1726.jpg?hmac=u0_XUn-JRaNrL-9fSm-m87xL3JtQbHFxQ068EpJSgb4"},
    {
        "text": "The aurora borealis over a snowy mountain",
        "image": "https://fastly.picsum.photos/id/901/3517/1726.jpg?hmac=u0_XUn-JRaNrL-9fSm-m87xL3JtQbHFxQ068EpJSgb4",
    },
    {"text": "A skyscraper high above the other buildings in a city on a cloudy day."},
    {"image": "https://fastly.picsum.photos/id/898/2655/1331.jpg?hmac=grTVBjfqQmnPY63ZCi1h82RC1Q1rDfGSmpSJSjfzIjU"},
    {
        "text": "A skyscraper high above the other buildings in a city on a cloudy day.",
        "image": "https://fastly.picsum.photos/id/898/2655/1331.jpg?hmac=grTVBjfqQmnPY63ZCi1h82RC1Q1rDfGSmpSJSjfzIjU",
    },
    {"text": "The top of a tower with an antenna on an overcast day."},
    {"image": "https://fastly.picsum.photos/id/500/2960/1555.jpg?hmac=lWAHvok_5yk5PpJwOxgU-bLEr4gPAHoXrJlkmZdkl_I"},
    {
        "text": "The top of a tower with an antenna on an overcast day.",
        "image": "https://fastly.picsum.photos/id/500/2960/1555.jpg?hmac=lWAHvok_5yk5PpJwOxgU-bLEr4gPAHoXrJlkmZdkl_I",
    },
    {"text": "Black and white image of the middle of the statue of liberty."},
    {"image": "https://fastly.picsum.photos/id/742/3784/1140.jpg?hmac=AzDecEd-uYZFG4vVKpP9XY17gY7TjRdKs5iQn5LxIn8"},
    {
        "text": "Black and white image of the middle of the statue of liberty.",
        "image": "https://fastly.picsum.photos/id/742/3784/1140.jpg?hmac=AzDecEd-uYZFG4vVKpP9XY17gY7TjRdKs5iQn5LxIn8",
    },
    {"text": "Black and white image of a lake reflecting the trees by its side."},
    {"image": "https://fastly.picsum.photos/id/412/3630/1502.jpg?hmac=Cg4GcGfWz7q3cI-Cf9Sxfrx2j75BzYGsHZgPDdH-ns8"},
    {
        "text": "Black and white image of a lake reflecting the trees by its side.",
        "image": "https://fastly.picsum.photos/id/412/3630/1502.jpg?hmac=Cg4GcGfWz7q3cI-Cf9Sxfrx2j75BzYGsHZgPDdH-ns8",
    },
    {"text": "A man crouching staring down at the tops of clouds from a mountain."},
    {"image": "https://fastly.picsum.photos/id/685/2853/1335.jpg?hmac=X4eZPprxEVmxX--D-0yNI235iDLFdn9ifMhQKNNX4vU"},
    {
        "text": "A man crouching staring down at the tops of clouds from a mountain.",
        "image": "https://fastly.picsum.photos/id/685/2853/1335.jpg?hmac=X4eZPprxEVmxX--D-0yNI235iDLFdn9ifMhQKNNX4vU",
    },
]

text_only = all_inputs[::3]
image_only = all_inputs[1::3]
joint_text_image = all_inputs[2::3]

multiimage_data = [
    {"text": "Images of skyscrapers high in the clouds"},
    {
        "image": [
            "https://fastly.picsum.photos/id/898/2655/1331.jpg?hmac=grTVBjfqQmnPY63ZCi1h82RC1Q1rDfGSmpSJSjfzIjU",
            "https://fastly.picsum.photos/id/500/2960/1555.jpg?hmac=lWAHvok_5yk5PpJwOxgU-bLEr4gPAHoXrJlkmZdkl_I",
        ]
    },
    {
        "text": "Images of skyscrapers high in the clouds",
        "image": [
            "https://fastly.picsum.photos/id/898/2655/1331.jpg?hmac=grTVBjfqQmnPY63ZCi1h82RC1Q1rDfGSmpSJSjfzIjU",
            "https://fastly.picsum.photos/id/500/2960/1555.jpg?hmac=lWAHvok_5yk5PpJwOxgU-bLEr4gPAHoXrJlkmZdkl_I",
        ],
    },
]

base_prompt = "Given a search query, retrieve relevant candidates that answer the query."

try:
    script_path = Path(__file__).parent.resolve()
except Exception:
    script_path = Path("./")
