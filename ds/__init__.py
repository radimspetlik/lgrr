

def update_db():
    import requests
    from tqdm import tqdm
    import base64

    api_data = {'hash': 'asldkf98732lka9823098(*%^'}

    res = requests.post('https://helpresearch.eu/api/list_images', json=api_data)

    if not res.ok:
        print(res.reason)
        return

    server_db = res.json()['data']

    import os
    from PIL import Image
    from io import BytesIO
    dataset_dir = os.path.join('/', os.sep, 'home.dokt', 'spetlrad', 'datagrid', 'reflections', 'db', 'lamps')
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)

    for user_id in tqdm(server_db.keys()):
        image_dir = os.path.join(dataset_dir, '{:07d}'.format(int(user_id)))
        if not os.path.isdir(image_dir):
            os.makedirs(image_dir)
        for image_id in server_db[user_id]:
            image_filepath = os.path.join(image_dir, '{:07d}.png'.format(int(image_id)))
            if not os.path.isfile(image_filepath):
                api_data = {'hash': 'asldkf98732lka9823098(*%^',
                            'image_id': image_id,
                            'user_id': user_id}
                res = requests.post('https://helpresearch.eu/api/get_image', json=api_data)

                str_im = res.json()['image']

                file = base64.b64decode(str_im.encode('utf8'))
                im = Image.open(BytesIO(file))
                im.save(image_filepath)

if __name__ == '__main__':
    update_db()
