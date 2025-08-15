from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from .model_loader import predict_image


def landing_page(request):
    return render(request, 'index.html')


def classify_image(request):
    context = {}
    if request.method == 'POST' and request.FILES.get('image'):
        image = request.FILES['image']
        fs = FileSystemStorage()
        saved_path = fs.save(image.name, image)          # saves file in MEDIA_ROOT
        full_path = fs.path(saved_path)                  # full absolute path to the file

        prediction = predict_image(full_path)            # call with full path
        context['prediction'] = prediction
        context['image_url'] = fs.url(saved_path)

    return render(request, 'classify.html', context)


