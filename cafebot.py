import cv2, traceback, glob, os, numpy as np, time, glitch_this, cv2.data, random, math, shutil, hikari, lightbulb
from scipy.interpolate import UnivariateSpline
from os.path import exists
from bing_image_downloader import downloader
from PIL import Image, ImageFilter
starter = glitch_this.ImageGlitcher()

bot = lightbulb.BotApp("")

@bot.listen(hikari.StartedEvent)
async def on_ready(event):
    print("Ready!")

@bot.command
@lightbulb.option("image", "The image to search for.", required=True)
@lightbulb.command("download", "Download an image to start.")
@lightbulb.implements(lightbulb.SlashCommand)
async def imagedownloader(ctx: lightbulb.context.Context):
    try:
        if exists(f"originalimage-{ctx.user.id}.png"):
            await ctx.respond("Existing image detected! Removing...")
            os.remove(f"originalimage-{ctx.user.id}.png")
        else:
            downloader.download(ctx.options.image, limit=1, output_dir="./", adult_filter_off=True)
            os.chdir(f"{ctx.options.image}/")
            os.chdir("..")
            for file in glob.glob(f"{ctx.options.image}/*.png"):
                if exists(file):
                    os.rename(f"{file}", f"originalimage-{ctx.user.id}.png")
                os.rmdir(f"{ctx.options.image}/")
            for file2 in glob.glob(f"{ctx.options.image}/*.jpg"):
                if exists(file2):
                    os.rename(f"{file2}", f"originalimage-{ctx.user.id}.png")
                os.rmdir(f"{ctx.options.image}/")
            await ctx.respond("Image downloaded!")
            imgname = f"originalimage-{ctx.user.id}.png"
            img = Image.open(imgname)
            fixed = 800
            height_percent = (fixed / float(img.size[1]))
            width_size = int((float(img.size[0]) * float(height_percent)))
            img = img.resize((width_size, fixed), Image.Resampling.LANCZOS)
            os.remove(imgname)
            time.sleep(0.50)
            img.save(imgname, "PNG", quality=100)
            await ctx.respond("Image downloaded!")
            f = hikari.File(imgname)
            await ctx.respond(f)
            return
    except:
        await ctx.respond("Error! Something went wrong." + traceback.format_exc())
        return

@bot.command
@lightbulb.command("blackandwhite", "Applies black and white filter.")
@lightbulb.implements(lightbulb.SlashCommand)
async def blackandwhite(ctx: lightbulb.context.Context):
    try:
        originalimage = cv2.imread(f"originalimage-{ctx.user.id}.png")
        grayimg = cv2.cvtColor(originalimage, cv2.COLOR_BGR2GRAY)
        (thresh, blackandwhiteimage) = cv2.threshold(grayimg, 127, 255, cv2.THRESH_BINARY)
        cv2.imwrite(f"blackandwhite-{ctx.user.id}.png", blackandwhiteimage)
        await ctx.respond("Black and white filter applied!")
        f = hikari.File(f"blackandwhite-{ctx.user.id}.png")
        await ctx.respond(f)
        time.sleep(1)
        os.remove(f"blackandwhite-{ctx.user.id}.png")
    except:
        await ctx.respond("Error! Something went wrong." + traceback.format_exc())

@bot.command
@lightbulb.option("alt_method", "Weither to use alt cartoon filter method.", required=True, type=bool)
@lightbulb.command("cartoon", "Applies cartoon filter.")
@lightbulb.implements(lightbulb.SlashCommand)
async def cartoon(ctx: lightbulb.context.Context):
    try:
        if ctx.options.alt_method == True:
            img = cv2.imread(f"originalimage-{ctx.user.id}.png")
            grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            grayimg = cv2.GaussianBlur(grayimg, (3, 3), 0)
            edgeimg = cv2.Laplacian(grayimg, -1, ksize=5)
            edgeimg = 255 - edgeimg
            ret, edgeimg = cv2.threshold(edgeimg, 150, 255, cv2.THRESH_BINARY)
            edgepreserve = cv2.edgePreservingFilter(img, flags=2, sigma_s=50, sigma_r=0.4)
            output = np.zeros(grayimg.shape)
            output = cv2.bitwise_and(edgepreserve, edgepreserve, mask=edgeimg)
            cv2.imwrite(f"cartoon_alt-{ctx.user.id}.png", output)
            await ctx.respond("Cartoon filter applied!")
            f = hikari.File(f"cartoon_alt-{ctx.user.id}.png")
            await ctx.respond(f)
            time.sleep(1)
            os.remove(f"cartoon_alt-{ctx.user.id}.png")
        else:
            originalimage = cv2.imread(f"originalimage-{ctx.user.id}.png")
            line_size = 7
            blur_value = 7
            k = 9
            d = 7
            gray = cv2.cvtColor(originalimage, cv2.COLOR_BGR2GRAY)
            gray_blur = cv2.medianBlur(gray, blur_value)
            edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_size, blur_value)
            data = np.float32(originalimage).reshape((-1, 3))
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
            ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            center = np.uint8(center)
            result = center[label.flatten()]
            result = result.reshape((originalimage.shape))
            blurred = cv2.bilateralFilter(result, d, sigmaColor=200, sigmaSpace=200)
            cartoon = cv2.bitwise_and(blurred, blurred, mask=edges)
            cv2.imwrite(f"cartoon-{ctx.user.id}.png", cartoon)
            await ctx.respond("Cartoon filter applied!")
            f = hikari.File(f"cartoon-{ctx.user.id}.png")
            await ctx.respond(f)
            time.sleep(1)
            os.remove(f"cartoon-{ctx.user.id}.png")
    except:
        await ctx.respond("Error! Something went wrong." + traceback.format_exc())

@bot.command
@lightbulb.command("invert", "Applies an invert filter.")
@lightbulb.implements(lightbulb.SlashCommand)
async def invert(ctx: lightbulb.context.Context):
    try:
        originalimage = cv2.imread(f"originalimage-{ctx.user.id}.png", 0)
        invertedimg = cv2.bitwise_not(originalimage)
        cv2.imwrite(f"inverted-{ctx.user.id}.png", invertedimg)
        await ctx.respond("Inverted image created!")
        f = hikari.File(f"inverted-{ctx.user.id}.png")
        await ctx.respond(f)
        time.sleep(1)
        os.remove(f"inverted-{ctx.user.id}.png")
    except:
        await ctx.respond("Error! Something went wrong." + traceback.format_exc())

@bot.command
@lightbulb.command("pencil_sketch", "Applies a pencil sketch filter.")
@lightbulb.implements(lightbulb.SlashCommand)
async def pencilsketch(ctx: lightbulb.context.Context):
    try:
        img = cv2.imread(f"originalimage-{ctx.user.id}.png")
        gray, color = cv2.pencilSketch(img, sigma_s=40, sigma_r=0.15, shade_factor=0.06)
        cv2.imwrite(f"pencilsketch-{ctx.user.id}.png", gray)
        await ctx.respond("Pencil sketch filter applied!")
        f = hikari.File(f"pencilsketch-{ctx.user.id}.png")
        await ctx.respond(f)
        time.sleep(1)
        os.remove(f"pencilsketch-{ctx.user.id}.png")
    except:
        await ctx.respond("Error! Something went wrong." + traceback.format_exc())

@bot.command
@lightbulb.command("sepia", "Applies a sepia filter.")
@lightbulb.implements(lightbulb.SlashCommand)
async def sepiaimg(ctx: lightbulb.context.Context):
    try:
        img = cv2.imread(f"originalimage-{ctx.user.id}.png")
        sepia = np.array(img, dtype=np.float64)
        sepia = cv2.transform(sepia, np.matrix([[0.272, 0.534, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]]))
        sepia[np.where(sepia > 255)] = 255
        sepia = np.array(sepia, dtype=np.uint8)
        cv2.imwrite(f"sepia-{ctx.user.id}.png", sepia)
        await ctx.respond("Sepia filter applied!")
        f = hikari.File(f"sepia-{ctx.user.id}.png")
        await ctx.respond(f)
        time.sleep(1)
        os.remove(f"sepia-{ctx.user.id}.png")
    except:
        await ctx.respond("Error! Something went wrong." + traceback.format_exc())

@bot.command
@lightbulb.option("num_blur", "Amount to blur.", required=True, type=int)
@lightbulb.command("blur", "Applies blur.")
@lightbulb.implements(lightbulb.SlashCommand)
async def blur(ctx: lightbulb.context.Context):
    try:
        img = Image.open(f"originalimage-{ctx.user.id}.png")
        numbtoblur = ctx.options.num_blur
        img = img.filter(ImageFilter.GaussianBlur(numbtoblur))
        img.save(f"blur_{numbtoblur}-{ctx.user.id}.png")
        await ctx.respond("Blur applied!")
        f = hikari.File(f"blur_{numbtoblur}-{ctx.user.id}.png")
        await ctx.respond(f)
        time.sleep(1)
        os.remove(f"blur_{numbtoblur}-{ctx.user.id}.png")
    except:
        await ctx.respond("Error! Something went wrong." + traceback.format_exc())

@bot.command
@lightbulb.command("contour", "Finds/draws contours in the image.")
@lightbulb.implements(lightbulb.SlashCommand)
async def contour(ctx: lightbulb.context.Context):
    try:
        img = cv2.imread(f"originalimage-{ctx.user.id}.png")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (0, 255, 0), 2, lineType=cv2.LINE_AA)
        cv2.imwrite(f"contour-{ctx.user.id}.png", img)
        await ctx.respond("Contours applied!")
        f = hikari.File(f"contour-{ctx.user.id}.png")
        await ctx.respond(f)
        time.sleep(1)
        os.remove(f"contour-{ctx.user.id}.png")
    except:
        await ctx.respond("Error! Something went wrong." + traceback.format_exc())

@bot.command
@lightbulb.command("detail", "Adds more detail to an image.")
@lightbulb.implements(lightbulb.SlashCommand)
async def detailed(ctx: lightbulb.context.Context):
    try:
        img = Image.open(f"originalimage-{ctx.user.id}.png")
        img = img.filter(ImageFilter.DETAIL)
        img.save(f"detail-{ctx.user.id}.png")
        await ctx.respond("Detail applied!")
        f = hikari.File(f"detail-{ctx.user.id}.png")
        await ctx.respond(f)
        time.sleep(1)
        os.remove(f"detail-{ctx.user.id}.png")
    except:
        await ctx.respond("Error! Something went wrong." + traceback.format_exc())

@bot.command
@lightbulb.command("edge_enhance", "Enhances the edges.")
@lightbulb.implements(lightbulb.SlashCommand)
async def edgeenhance(ctx: lightbulb.context.Context):
    try:
        img = Image.open(f"originalimage-{ctx.user.id}.png")
        img = img.convert("RGB")
        img = img.filter(ImageFilter.EDGE_ENHANCE)
        img.save(f"edgeenhanced-{ctx.user.id}.png")
        await ctx.respond("Edges enhanced!")
        f = hikari.File(f"edgeenhanced-{ctx.user.id}.png")
        await ctx.respond(f)
        time.sleep(1)
        os.remove(f"edgeenhanced-{ctx.user.id}.png")
    except:
        await ctx.respond("Error! Something went wrong." + traceback.format_exc())

@bot.command
@lightbulb.command("emboss", "Emboss the edges.")
@lightbulb.implements(lightbulb.SlashCommand)
async def emboss(ctx: lightbulb.context.Context):
    try:
        img = Image.open(f"originalimage-{ctx.user.id}.png")
        img = img.convert("RGB")
        img = img.filter(ImageFilter.EMBOSS)
        img.save(f"emboss-{ctx.user.id}.png")
        await ctx.respond("Emboss added!")
        f = hikari.File(f"emboss-{ctx.user.id}.png")
        await ctx.respond(f)
        time.sleep(1)
        os.remove(f"emboss-{ctx.user.id}.png")
    except:
        await ctx.respond("Error! Something went wrong." + traceback.format_exc())

@bot.command
@lightbulb.command("edge_finder", "Finds edges.")
@lightbulb.implements(lightbulb.SlashCommand)
async def edgefinder(ctx: lightbulb.context.Context):
    try:
        img = Image.open(f"originalimage-{ctx.user.id}.png")
        img = img.convert("RGB")
        img = img.filter(ImageFilter.FIND_EDGES)
        img.save(f"edgefinder-{ctx.user.id}.png")
        await ctx.respond("Found the edges!")
        f = hikari.File(f"edgefinder-{ctx.user.id}.png")
        await ctx.respond(f)
        time.sleep(1)
        os.remove(f"edgefinder-{ctx.user.id}.png")
    except:
        await ctx.respond("Error! Something went wrong." + traceback.format_exc())

@bot.command
@lightbulb.command("sharpen", "Sharpens the image.")
@lightbulb.implements(lightbulb.SlashCommand)
async def sharpen(ctx: lightbulb.context.Context):
    try:
        img = Image.open(f"originalimage-{ctx.user.id}.png")
        img = img.convert("RGB")
        img = img.filter(ImageFilter.SHARPEN)
        img.save(f"sharpen-{ctx.user.id}.png")
        await ctx.respond("Image sharpened!")
        f = hikari.File(f"sharpen-{ctx.user.id}.png")
        await ctx.respond(f)
        time.sleep(1)
        os.remove(f"sharpen-{ctx.user.id}.png")
    except:
        await ctx.respond("Error! Something went wrong." + traceback.format_exc())

@bot.command
@lightbulb.command("edge_enhance", "Enhances the edges.")
@lightbulb.implements(lightbulb.SlashCommand)
async def edgeenhance(ctx: lightbulb.context.Context):
    try:
        img = Image.open(f"originalimage-{ctx.user.id}.png")
        img = img.convert("RGB")
        img = img.filter(ImageFilter.SMOOTH_MORE)
        img.save(f"smooth-{ctx.user.id}.png")
        await ctx.respond("Smoothed!")
        f = hikari.File(f"smooth-{ctx.user.id}.png")
        await ctx.respond(f)
        time.sleep(1)
        os.remove(f"smooth-{ctx.user.id}.png")
    except:
        await ctx.respond("Error! Something went wrong." + traceback.format_exc())

@bot.command
@lightbulb.command("summer", "Makes a summer filter, similar to Instagram.")
@lightbulb.implements(lightbulb.SlashCommand)
async def summer(ctx: lightbulb.context.Context):
    try:
        img = cv2.imread(f"originalimage-{ctx.user.id}.png")
        def table(x, y):
            spline = UnivariateSpline(x, y)
            return spline(range(256))
        increase = table([0, 64, 128, 256], [0, 80, 160, 256])
        decrease = table([0, 64, 128, 256], [0, 50, 100, 256])
        blue, green, red = cv2.split(img)
        red = cv2.LUT(red, increase).astype(np.uint8)
        blue = cv2.LUT(blue, decrease).astype(np.uint8)
        sum = cv2.merge((blue, green, red))
        cv2.imwrite(f"summer-{ctx.user.id}.png", sum)
        await ctx.respond("Summer filter applied!")
        f = hikari.File(f"summer-{ctx.user.id}.png")
        await ctx.respond(f)
        time.sleep(1)
        os.remove(f"summer-{ctx.user.id}.png")
    except:
        await ctx.respond("Error! Something went wrong." + traceback.format_exc())

@bot.command
@lightbulb.command("winter", "Makes a winter filter similar to Instagram.")
@lightbulb.implements(lightbulb.SlashCommand)
async def winter(ctx: lightbulb.context.Context):
    try:
        img = cv2.imread(f"originalimage-{ctx.user.id}.png")
        def table(x, y):
            spline = UnivariateSpline(x, y)
            return spline(range(256))
        increase = table([0, 64, 128, 256], [0, 80, 160, 256])
        decrease = table([0, 64, 128, 256], [0, 50, 100, 256])
        blue, green, red = cv2.split(img)
        red = cv2.LUT(red, decrease).astype(np.uint8)
        blue = cv2.LUT(blue, increase).astype(np.uint8)
        sum = cv2.merge((blue, green, red))
        cv2.imwrite(f"winter-{ctx.user.id}.png", sum)
        await ctx.respond("Winter filter applied!")
        f = hikari.File(f"winter-{ctx.user.id}.png")
        await ctx.respond(f)
        time.sleep(1)
        os.remove(f"winter-{ctx.user.id}.png")
    except:
        await ctx.respond("Error! Something went wrong." + traceback.format_exc())

@bot.command
@lightbulb.option("scanlines", "Weither to add scanlines.", required=True, type=bool)
@lightbulb.option("color_offset", "Weither to glitch the color offset.", required=True, type=bool)
@lightbulb.option("amount", "Amount of times to glitch the image.", required=True, type=int)
@lightbulb.command("glitch", "Glitches the image.")
@lightbulb.implements(lightbulb.SlashCommand)
async def glitch(ctx: lightbulb.context.Context):
    try:
        img = Image.open(f"originalimage-{ctx.user.id}.png")
        amt = ctx.options.amount
        coloroffset = ctx.options.color_offset
        scanlines = ctx.options.scanlines
        glitch = starter.glitch_image(img, amt, color_offset=coloroffset, scan_lines=scanlines)
        glitch.save(f"glitchimg_amt_{str(amt)}-{ctx.user.id}.png")
        await ctx.respond("Image glitched successfully!")
        f = hikari.File(f"glitchimg_amt_{str(amt)}-{ctx.user.id}.png")
        await ctx.respond(f)
        time.sleep(1)
        os.remove(f"glitchimg_amt_{str(amt)}-{ctx.user.id}.png")
    except:
        await ctx.respond("Error! Something went wrong." + traceback.format_exc())

@bot.command
@lightbulb.command("ascii", "Transforms the image into ascii.")
@lightbulb.implements(lightbulb.SlashCommand)
async def asciiart(ctx: lightbulb.context.Context):
    try:
        await ctx.respond("This mode is very experimental!!\n\nDue to the way hikari works, the ascii result will need to be sent as a txt file.\n\n**The command will continue to run after 5 seconds.**")
        time.sleep(5)
        img = Image.open(f"originalimage-{ctx.user.id}.png")
        width, height = img.size
        ratio = height / width
        new_width = 120
        new_height = ratio * new_width * 0.55
        img = img.resize((new_width, int(new_height)))
        img = img.convert('L')
        chars = ["@", "#", "$", "%", "?", "*", "+", ";", ":", ",", "."]
        pixels = img.getdata()
        new_pixels = [chars[pixels//25] for pixels in pixels]
        new_pixels = ''.join(new_pixels)
        new_pixels_count = len(new_pixels)
        ascii_image = [new_pixels[index:index + new_width] for index in range(0, new_pixels_count, new_width)]
        ascii_image = "\n".join(ascii_image)
        with open(f"asciiart-{ctx.user.id}.txt", 'w') as f:
            f.write(ascii_image)
        await ctx.edit_last_response("ASCII file created!")
        f = hikari.File(f"asciiart-{ctx.user.id}.txt")
        await ctx.respond(f)
        time.sleep(1)
        os.remove(f"asciiart-{ctx.user.id}.txt")
    except:
        await ctx.respond("Error! Something went wrong." + traceback.format_exc())

@bot.command
@lightbulb.command("moon", "Applies moon filter.")
@lightbulb.implements(lightbulb.SlashCommand)
async def moon(ctx: lightbulb.context.Context):
    try:
        img = cv2.imread(f"originalimage-{ctx.user.id}.png")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.equalizeHist(img)
        cv2.imwrite(f"moon-{ctx.user.id}.png", img)
        await ctx.respond("Moon filter added!")
        f = hikari.File(f"moon-{ctx.user.id}.png")
        await ctx.respond(f)
        time.sleep(1)
        os.remove(f"moon-{ctx.user.id}.png")
    except:
        await ctx.respond("Error! Something went wrong." + traceback.format_exc())

@bot.command
@lightbulb.command("clarendon", "Applies clarendon filter.")
@lightbulb.implements(lightbulb.SlashCommand)
async def clarendon(ctx: lightbulb.context.Context):
    try:
        img = cv2.imread(f"originalimage-{ctx.user.id}.png")
        clarendon = img.copy()
        blue, green, red = cv2.split(clarendon)
        ogvalues = np.array([0, 28, 56, 85, 113, 141, 170, 198, 227, 255])
        blueval = np.array([0, 38, 66, 104, 139, 175, 206, 226, 245, 255])
        redval = np.array([0, 16, 35, 64, 117, 163, 200, 222, 237, 249])
        greenval = np.array([0, 24, 49, 98, 141, 174, 201, 223, 239, 255])
        fullrange = np.arange(0, 256)
        bluelookup = np.interp(fullrange, ogvalues, blueval)
        greenlookup = np.interp(fullrange, ogvalues, greenval)
        redlookup = np.interp(fullrange, ogvalues, redval)
        bluechannel = cv2.LUT(blue, bluelookup)
        greenchannel = cv2.LUT(green, greenlookup)
        redchannel = cv2.LUT(red, redlookup)
        clarendon = cv2.merge([bluechannel, greenchannel, redchannel])
        clarendon = np.uint8(clarendon)
        cv2.imwrite(f"clarendon-{ctx.user.id}.png", clarendon)
        await ctx.respond("Clarendon filter applied!")
        f = hikari.File(f"clarendon-{ctx.user.id}.png")
        await ctx.respond(f)
        time.sleep(1)
        os.remove(f"clarendon-{ctx.user.id}.png")
    except:
        await ctx.respond("Error! Something went wrong." + traceback.format_exc())

@bot.command
@lightbulb.command("laplacian", "Applies laplacian filter.")
@lightbulb.implements(lightbulb.SlashCommand)
async def laplacian(ctx: lightbulb.context.Context):
    try:
        image = cv2.imread(f"originalimage-{ctx.user.id}.png")
        laplacian = cv2.Laplacian(image, cv2.CV_32F, ksize=3, scale=1, delta=0)
        logkernel = np.array(([0.4038, 0.8021, 0.4038], [0.8021, -4.8233, 0.8021], [0.4038, 0.8021, 0.4038]), dtype="float")
        logimg = cv2.filter2D(image, cv2.CV_32F, logkernel)
        cv2.normalize(laplacian, laplacian, alpha=0, beta=1, normType=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        cv2.normalize(logimg, logimg, alpha=0, beta=1, normType=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        cv2.imwrite(f"laplacian-{ctx.user.id}.png", laplacian)
        await ctx.respond("Laplacian filter applied!")
        f = hikari.File(f"laplacian-{ctx.user.id}.png")
        await ctx.respond(f)
        time.sleep(1)
        os.remove(f"lablacian-{ctx.user.id}.png")
    except:
        await ctx.respond("Error! Something went wrong." + traceback.format_exc())

@bot.command
@lightbulb.command("kelvin", "Applies kelvin filter.")
@lightbulb.implements(lightbulb.SlashCommand)
async def kelvin(ctx: lightbulb.context.Context):
    try:
        img = cv2.imread(f"originalimage-{ctx.user.id}.png")
        output = img.copy()
        bluechannel, greenchannel, redchannel = cv2.split(output)
        redValuesOriginal = np.array([0, 60, 110, 150, 235, 255])
        redValues = np.array([0, 102, 185, 220, 245, 245 ])
        greenValuesOriginal = np.array([0, 68, 105, 190, 255])
        greenValues = np.array([0, 68, 120, 220, 255 ])
        blueValuesOriginal = np.array([0, 88, 145, 185, 255])
        blueValues = np.array([0, 12, 140, 212, 255])
        allvalues = np.arange(0, 256)
        bluelookup = np.interp(allvalues, blueValuesOriginal, blueValues)
        greenlookup = np.interp(allvalues, greenValuesOriginal, greenValues)
        redlookup = np.interp(allvalues, redValuesOriginal, redValues)
        bluechannel = cv2.LUT(bluechannel, bluelookup)
        greenchannel = cv2.LUT(greenchannel, greenlookup)
        redchannel = cv2.LUT(redchannel, redlookup)
        output = cv2.merge([bluechannel, greenchannel, redchannel])
        output = np.uint8(output)
        cv2.imwrite(f"kelvin-{ctx.user.id}.png", output)
        await ctx.respond("Kelvin filter applied!")
        f = hikari.File(f"kelvin-{ctx.user.id}.png")
        await ctx.respond(f)
        time.sleep(1)
        os.remove(f"kelvin-{ctx.user.id}.png")
    except:
        await ctx.respond("Error! Something went wrong." + traceback.format_exc())

@bot.command
@lightbulb.command("xpro", "Applies xpro filter.")
@lightbulb.implements(lightbulb.SlashCommand)
async def xpro(ctx: lightbulb.context.Context):
    try:
        img = cv2.imread(f"originalimage-{ctx.user.id}.png")
        output = img.copy()
        B, G, R = cv2.split(output)
        vignettescale = 6
        k = np.min([output.shape[1], output.shape[0]]) / vignettescale
        kernelx = cv2.getGaussianKernel(output.shape[1], k)
        kernely = cv2.getGaussianKernel(output.shape[0], k)
        kernel = kernely * kernelx.T
        mask = cv2.normalize(kernel, None, alpha=0, beta=1, normType=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        B = B + B * mask
        G = G + G * mask
        R = R + R * mask
        output = cv2.merge([B, G, R])
        output = output / 2
        output = np.clip(output, 0, 255)
        output = np.uint8(output)
        B, G, R = cv2.split(output)
        redvaluesoriginal = np.array([0, 42, 105, 148, 185, 255])
        redvalues = np.array([0, 28, 100, 165, 215, 255])
        greenvaluesoriginal = np.array([0, 40, 85, 125, 165, 212, 255])
        greenvalues = np.array([0, 25, 75, 135, 185, 230, 255])
        bluevaluesoriginal = np.array([0, 40, 82, 125, 170, 225, 255])
        bluevalues = np.array([0, 38, 90, 125, 160, 210, 222])
        allvalues = np.arange(0, 256)
        redlookup = np.interp(allvalues, redvaluesoriginal, redvalues)
        R = cv2.LUT(R, redlookup)
        greenlookup = np.interp(allvalues, greenvaluesoriginal, greenvalues)
        G = cv2.LUT(G, greenlookup)
        bluelookup = np.interp(allvalues, bluevaluesoriginal, bluevalues)
        B = cv2.LUT(B, bluelookup)
        output = cv2.merge([B, G, R])
        output = np.uint8(output)
        output = cv2.cvtColor(output, cv2.COLOR_BGR2YCrCb)
        output = np.float32(output)
        Y, Cr, Cb = cv2.split(output)
        Y = Y * 1.2
        Y = np.clip(Y, 0, 255)
        output = cv2.merge([Y, Cr, Cb])
        output = np.uint8(output)
        output = cv2.cvtColor(output, cv2.COLOR_YCrCb2BGR)
        cv2.imwrite(f"xpro-{ctx.user.id}.png", output)
        await ctx.respond("Xpro filter applied!")
        f = hikari.File(f"xpro-{ctx.user.id}.png")
        await ctx.respond(f)
        time.sleep(1)
        os.remove(f"xpro-{ctx.user.id}.png")
    except:
        await ctx.respond("Error! Something went wrong." + traceback.format_exc())

@bot.command
@lightbulb.option("erode_amt", "Amount to erode the image.", required=True, type=int)
@lightbulb.command("erode", "Erodes the image.")
@lightbulb.implements(lightbulb.SlashCommand)
async def erode(ctx: lightbulb.context.Context):
    try:
        img = cv2.imread(f"originalimage-{ctx.user.id}.png")
        erosionsize = ctx.options.erode_amt
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (2 * erosionsize + 1, 2 * erosionsize + 1), (erosionsize, erosionsize))
        erodeimg = cv2.erode(img, element)
        cv2.imwrite(f"erode-{ctx.user.id}.png", erodeimg)
        await ctx.respond("Image eroded successfully!")
        f = hikari.File(f"erode-{ctx.user.id}.png")
        await ctx.respond(f)
        time.sleep(1)
        os.remove(f"erode-{ctx.user.id}.png")
    except:
        await ctx.respond("Error! Something went wrong." + traceback.format_exc())

bot.run(
    activity=hikari.Activity(
        name="with photos.",
        type=hikari.ActivityType.PLAYING
    ),
    ignore_session_start_limit=True,
    check_for_updates=False,
    status=hikari.Status.ONLINE,
    coroutine_tracking_depth=20,
    propagate_interrupts=True
)