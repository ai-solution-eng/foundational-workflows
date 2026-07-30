import base64
import mimetypes


def save_data_url_with_auto_extension(data_url: str, base_filename: str):
    header, b64_data = data_url.split(",", 1)

    # Extract the mime type (e.g., "image/png") from "data:image/png;base64"
    mime_type = header.split(";")[0].split(":")[1]

    # Guess the extension (e.g., ".png")
    extension = mimetypes.guess_extension(mime_type) or ".jpg"

    # Create the full output path
    output_path = f"{base_filename}{extension}"

    # Decode and save
    image_bytes = base64.b64decode(b64_data)
    with open(output_path, "wb") as f:
        f.write(image_bytes)

    print(f"Saved image successfully to {output_path}")
