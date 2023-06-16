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

 @bot.command
@lightbulb.option("dilate_amt", "Amount to dilate the image.", required=True, type=int)
@lightbulb.command("dilate", "Dilates the image.")
@lightbulb.implements(lightbulb.SlashCommand)
async def dilate(ctx: lightbulb.context.Context):
    try:
        img = cv2.imread("originalimage.png")
        dilateamt = ctx.options.dilate_amt
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (2 * dilateamt + 1, 2 * dilateamt + 1), (dilateamt, dilateamt))
        dilateimg = cv2.dilate(img, element)
        cv2.imwrite(f"dilate-{ctx.user.id}.png", dilateimg)
        await ctx.respond("Dilated successfully!")
        f = hikari.File(f"dilate-{ctx.user.id}.png")
        await ctx.respond(f)
        time.sleep(1)
        os.remove(f"dilate-{ctx.user.id}.png")
    except:
    	await ctx.respond("Error! Something went wrong." + traceback.format_exc())

@bot.command
@lightbulb.option("gamma_amt", "Amount of gamma to add.", required=True, type=int)
@lightbulb.command("gamma", "Adds gamma to the image.")
@lightbulb.implements(lightbulb.SlashCommand)
async def gamma(ctx: lightbulb.context.Context):
    try:
        img = cv2.imread(f"originalimage-{ctx.user.id}.png")
        gammaamt = ctx.options.gamma_amt
        values = np.arange(0, 256)
        lut = np.uint8(255 * np.power((values / 255.0), gammaamt))
        result = cv2.LUT(img, lut)
        cv2.imwrite(f"gamma-{ctx.user.id}.png", result)
        await ctx.respond("Gamma applied!")
        f = hikari.File(f"gamma-{ctx.user.id}.png")
        await ctx.respond(f)
        time.sleep(1)
        os.remove(f"gamma-{ctx.user.id}.png")
    except:
        await ctx.respond("Error! Something went wrong." + traceback.format_exc())

@bot.command
@lightbulb.option("contrast_amt", "How much contrast to apply.", required=True, type=int)
@lightbulb.command("contrast", "Contrasts the image.")
@lightbulb.implements(lightbulb.SlashCommand)
async def contrast(ctx: lightbulb.context.Context):
    try:
        img = cv2.imread(f"originalimage-{ctx.user.id}.png")
        imgycb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        imgycb = np.float32(imgycb)
        Y, C, B = cv2.split(imgycb)
        alpha = ctx.options.contrast_amt
        Y = Y * alpha
        Y = np.clip(Y, 0, 255)
        imgycb = cv2.merge([Y, C, B])
        imgycb = np.uint8(imgycb)
        result = cv2.cvtColor(imgycb, cv2.COLOR_YCrCb2BGR)
        cv2.imwrite(f"contrast-{ctx.user.id}.png", result)
        await ctx.respond("Contrast applied!")
        f = hikari.File(f"contrast-{ctx.user.id}.png")
        await ctx.respond(f)
        time.sleep(1)
        os.remove(f"contrast-{ctx.user.id}.png")
    except:
        await ctx.respond("Error! Something went wrong." + traceback.format_exc())

@bot.command
@lightbulb.option("bright_amt", "How much brightness to apply.", required=True, type=int)
@lightbulb.command("brightness", "Brightens the image.")
@lightbulb.implements(lightbulb.SlashCommand)
async def brightness(ctx: lightbulb.context.Context):
    try:
        img = cv2.imread(f"originalimage-{ctx.user.id}.png")
        imgycb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        imgycb = np.float32(imgycb)
        Y, C, B = cv2.split(imgycb)
        alpha = ctx.options.bright_amt
        Y = Y + alpha
        Y = np.clip(Y, 0, 255)
        imgycb = cv2.merge([Y, C, B])
        imgycb = np.uint8(imgycb)
        result = cv2.cvtColor(imgycb, cv2.COLOR_YCrCb2BGR)
        cv2.imwrite(f"brightness-{ctx.user.id}.png", result)
        await ctx.respond("Brightness applied!")
        f = hikari.File(f"brightness-{ctx.user.id}.png")
        await ctx.respond(f)
        time.sleep(1)
        os.remove(f"brightness-{ctx.user.id}.png")
    except:
        await ctx.respond("Error! Something went wrong." + traceback.format_exc())

@bot.command
@lightbulb.command("hsv", "Converts the image into HSV.")
@lightbulb.implements(lightbulb.SlashCommand)
async def hsvfilter(ctx: lightbulb.context.Context):
    try:
        img = cv2.imread(f"originalimage-{ctx.user.id}.png")
        hsvimg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsvimgcopy = hsvimg.copy()
        hsvimgcopy = np.float32(hsvimgcopy)
        saturationscale = 0.01
        H, S, V = cv2.split(hsvimgcopy)
        S = np.clip(S * saturationscale, 0, 255)
        hsvimgcopy = cv2.merge([H, S, V])
        hsvimgcopy = np.uint8(hsvimgcopy)
        hsvimgcopy = cv2.cvtColor(hsvimgcopy, cv2.COLOR_HSV2BGR)
        cv2.imwrite(f"hsv-{ctx.user.id}.png", hsvimgcopy)
        await ctx.respond("HSV applied")
        f = hikari.File(f"hsv-{ctx.user.id}.png")
        await ctx.respond(f)
        time.sleep(1)
        os.remove(f"hsv-{ctx.user.id}.png")
    except:
        await ctx.respond("Error! Something went wrong." + traceback.format_exc())

@bot.command
@lightbulb.option("rotate_amt", "How much rotation to apply.", required=True, type=int)
@lightbulb.command("rotate", "Rotates the image.")
@lightbulb.implements(lightbulb.SlashCommand)
async def rotate(ctx: lightbulb.context.Context):
    try:
        img = cv2.imread(f"originalimage-{ctx.user.id}.png")
        angle = ctx.options.rotate_amt
        rotation = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), angle, 1)
        result = cv2.warpAffine(img, rotation, (img.shape[1], img.shape[0]))
        cv2.imwrite(f"rotate-{ctx.user.id}.png", result)
        await ctx.respond("Image rotated!")
        f = hikari.File(f"rotate-{ctx.user.id}.png")
        await ctx.respond(f)
        time.sleep(1)
        os.remove(f"rotate-{ctx.user.id}.png")
    except:
        await ctx.respond("Error! Something went wrong." + traceback.format_exc())

@bot.command
@lightbulb.command("affine", "Affines the image.")
@lightbulb.implements(lightbulb.SlashCommand)
async def affine(ctx: lightbulb.context.Context):
    try:
        img = cv2.imread(f"originalimage-{ctx.user.id}.png")
        rows, cols, ch = img.shape
        pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
        pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
        M = cv2.getAffineTransform(pts1, pts2)
        dst = cv2.warpAffine(img, M, (cols, rows))
        cv2.imwrite(f"affine-{ctx.user.id}.png", dst)
        await ctx.respond("Affine applied!")
        f = hikari.File(f"contrast-{ctx.user.id}.png")
        await ctx.respond(f)
        time.sleep(1)
        os.remove(f"affine-{ctx.user.id}.png")
    except:
        await ctx.respond("Error! Something went wrong." + traceback.format_exc())

@bot.command
@lightbulb.command("inverse_affine", "Inverse affines the image.")
@lightbulb.implements(lightbulb.SlashCommand)
async def inveraffine(ctx: lightbulb.context.Context):
    try:
        img = cv2.imread(f"originalimage-{ctx.user.id}.png", cv2.IMREAD_GRAYSCALE)
        cv2.line(img, (450, 100), (750, 650), (0, 0, 255), 5, cv2.LINE_AA, 0)
        cv2.line(img, (750, 650), (1000, 300), (0, 0, 255), 5, cv2.LINE_AA, 0)
        cv2.line(img, (1000, 300), (450, 100), (0, 0, 255), 5, cv2.LINE_AA, 0)
        warpmat1 = np.float32([[1.2, 0.2, 2], [-0.3, 1.3, 1]])
        result1 = cv2.warpAffine(img, warpmat1, (int(1.5 * img.shape[1]), int(1.4 * img.shape[0])), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        cv2.imwrite(f"inverseaffine-{ctx.user.id}.png", result1)
        await ctx.respond("Inverse affine applied!")
        f = hikari.File(f"inverseaffine-{ctx.user.id}.png")
        await ctx.respond(f)
        time.sleep(1)
        os.remove(f"inverseaffine-{ctx.user.id}.png")
    except:
        await ctx.respond("Error! Something went wrong." + traceback.format_exc())

@bot.command
@lightbulb.command("saturate", "Saturates the image.")
@lightbulb.implements(lightbulb.SlashCommand)
async def saturate(ctx: lightbulb.context.Context):
    try:
        img = cv2.imread(f"originalimage-{ctx.user.id}.png")
        hsvimg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsvimg)
        cv2.imwrite(f"saturate-{ctx.user.id}.png", s)
        await ctx.respond("Saturation applied!")
        f = hikari.File(f"saturate-{ctx.user.id}.png")
        await ctx.respond(f)
        time.sleep(1)
        os.remove(f"saturate-{ctx.user.id}.png")
    except:
        await ctx.respond("Error! Something went wrong." + traceback.format_exc())

@bot.command
@lightbulb.command("hue", "Does stuff with the hue from the image.")
@lightbulb.implements(lightbulb.SlashCommand)
async def hue(ctx: lightbulb.context.Context):
    try:
        img = cv2.imread(f"originalimage-{ctx.user.id}.png")
        hsvimg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsvimg)
        cv2.imwrite(f"hue-{ctx.user.id}.png", h)
        await ctx.respond("Hue applied!")
        f = hikari.File(f"hue-{ctx.user.id}.png")
        await ctx.respond(f)
        time.sleep(1)
        os.remove(f"hue-{ctx.user.id}.png")
    except:
        await ctx.respond("Error! Something went wrong." + traceback.format_exc())

@bot.command
@lightbulb.command("cca", "Converts the image into CCA format.")
@lightbulb.implements(lightbulb.SlashCommand)
async def cca(ctx: lightbulb.context.Context):
    try:
        img = cv2.imread(f"originalimage-{ctx.user.id}.png", cv2.IMREAD_GRAYSCALE)
        th, binaryimg = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        _, binaryimg = cv2.connectedComponents(binaryimg)
        binaryimgclone = np.copy(binaryimg)
        (minval, maxval, minpos, maxpos) = cv2.minMaxLoc(binaryimgclone)
        binaryimgclone = 255 * (binaryimgclone - minval) / (maxval - minval)
        binaryimgclone = np.uint8(binaryimgclone)
        binimgclonecolormap = cv2.applyColorMap(binaryimgclone, cv2.COLORMAP_JET)
        cv2.imwrite(f"cca-{ctx.user.id}.png", binaryimgclone)
        await ctx.respond("CCA applied!")
        f = hikari.File(f"cca-{ctx.user.id}.png")
        await ctx.respond(f)
        time.sleep(1)
        os.remove(f"cca-{ctx.user.id}.png")
    except:
        await ctx.respond("Error! Something went wrong." + traceback.format_exc())

@bot.command
@lightbulb.command("desaturate", "Desaturates the image.")
@lightbulb.implements(lightbulb.SlashCommand)
async def desaturate(ctx: lightbulb.context.Context):
    try:
        img = cv2.imread(f"originalimage-{ctx.user.id}.png")
        hsvimg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsvimgcopy = hsvimg.copy()
        hsvimgcopy = np.float32(hsvimgcopy)
        scale = 0.01
        H, S, V = cv2.split(hsvimgcopy)
        S = np.clip(S * scale, 0, 255)
        hsvimgcopy = cv2.merge((H, S, V))
        hsvimgcopy = cv2.cvtColor(hsvimgcopy, cv2.COLOR_HSV2BGR)
        cv2.imwrite(f"desaturate-{ctx.user.id}.png", hsvimgcopy)
        await ctx.respond("Desaturation applied!")
        f = hikari.File(f"desaturate-{ctx.user.id}.png")
        await ctx.respond(f)
        time.sleep(1)
        os.remove(f"desaturate-{ctx.user.id}.png")
    except:
        await ctx.respond("Error! Something went wrong." + traceback.format_exc())

@bot.command
@lightbulb.option("thresh_amt", "Amount of threshold to apply.", required=True, type=int)
@lightbulb.command("threshold", "Applies threshold to the image.")
@lightbulb.implements(lightbulb.SlashCommand)
async def threshold(ctx: lightbulb.context.Context):
    try:
        img = cv2.imread(f"originalimage-{ctx.user.id}.png")
        threshamt = ctx.options.thresh_amt
        retval, threshold = cv2.threshold(img, threshamt, 255, cv2.THRESH_BINARY)
        cv2.imwrite(f"threshold-{ctx.user.id}.png", threshold)
        await ctx.respond("Threshold applied!")
        f = hikari.File(f"threshold-{ctx.user.id}.png")
        await ctx.respond(f)
        time.sleep(1)
        os.remove(f"threshold-{ctx.user.id}.png")
    except:
        await ctx.respond("Error! Something went wrong." + traceback.format_exc())

@bot.command
@lightbulb.command("solarize", "Solarizes the image.")
@lightbulb.implements(lightbulb.SlashCommand)
async def solarize(ctx: lightbulb.context.Context):
    try:
        img = cv2.imread(f"originalimage-{ctx.user.id}.png")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        imgcopy = img.copy()
        imgcopy = np.float32(imgcopy)
        H, S, V = cv2.split(imgcopy)
        V = np.clip(255 - V, 0, 255)
        imgcopy = cv2.merge((H, S, V))
        imgcopy = cv2.cvtColor(imgcopy, cv2.COLOR_HSV2BGR)
        cv2.imwrite(f"solarize-{ctx.user.id}.png", imgcopy)
        await ctx.respond("Solarize applied!")
        f = hikari.File(f"solarize-{ctx.user.id}.png")
        await ctx.respond(f)
        time.sleep(1)
        os.remove(f"solarize-{ctx.user.id}.png")
    except:
        await ctx.respond("Error! Something went wrong." + traceback.format_exc())

@bot.command
@lightbulb.option("pixel_amt", "Amount to apply pixelation.", required=True, type=int)
@lightbulb.command("pixelate", "Saturates the image.")
@lightbulb.implements(lightbulb.SlashCommand)
async def pixelate(ctx: lightbulb.context.Context):
    try:
        img = Image.open(f"originalimage-{ctx.user.id}.png")
        size = ctx.options.pixel_amt
        img = img.resize((img.size[0] // size, img.size[1] // size), Image.Resampling.NEAREST)
        img = img.resize((img.size[0] * size, img.size[1] * size), Image.Resampling.NEAREST)
        img.save(f"pixelated-{ctx.user.id}.png")
        await ctx.respond("Pixelation applied!")
        f = hikari.File(f"pixelation-{ctx.user.id}.png")
        await ctx.respond(f)
        time.sleep(1)
        os.remove(f"pixelation-{ctx.user.id}.png")
    except:
        await ctx.respond("Error! Something went wrong." + traceback.format_exc())

@bot.command
@lightbulb.command("oil_painting", "Applies an oil painting filter to the image.")
@lightbulb.implements(lightbulb.SlashCommand)
async def oilpainting(ctx: lightbulb.context.Context):
    try:
        img = cv2.imread(f"originalimage-{ctx.user.id}.png", cv2.IMREAD_GRAYSCALE)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
        morph = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        result = cv2.normalize(morph, None, 20, 255, cv2.NORM_MINMAX)
        cv2.imwrite(f"painting-{ctx.user.id}.png", result)
        await ctx.respond("Filter applied!")
        f = hikari.File(f"painting-{ctx.user.id}.png")
        await ctx.respond(f)
        time.sleep(1)
        os.remove(f"painting-{ctx.user.id}.png")
    except:
        await ctx.respond("Error! Something went wrong." + traceback.format_exc())

@bot.command
@lightbulb.command("posterize", "Posterize the image.")
@lightbulb.implements(lightbulb.SlashCommand)
async def posterize(ctx: lightbulb.context.Context):
    try:
        img = cv2.imread(f"originalimage-{ctx.user.id}.png")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        imgcopy = img.copy()
        imgcopy = np.float32(imgcopy)
        H, S, V = cv2.split(imgcopy)
        S = np.clip(S // 32 * 32, 0, 255)
        imgcopy = cv2.merge((H, S, V))
        imgcopy = cv2.cvtColor(imgcopy, cv2.COLOR_HSV2BGR)
        cv2.imwrite(f"contrast-{ctx.user.id}.png", imgcopy)
        await ctx.respond("Contrast applied!")
        f = hikari.File(f"contrast-{ctx.user.id}.png")
        await ctx.respond(f)
        time.sleep(1)
        os.remove(f"contrast-{ctx.user.id}.png")
    except:
        await ctx.respond("Error! Something went wrong." + traceback.format_exc())

@bot.command
@lightbulb.command("prewitt", "Applies prewitt to the image.")
@lightbulb.implements(lightbulb.SlashCommand)
async def prewitt(ctx: lightbulb.context.Context):
    try:
        img = cv2.imread(f"originalimage-{ctx.user.id}.png")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernelx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        imgx = cv2.filter2D(gray, -1, kernelx)
        absoul = np.absolute(imgx)
        normalize = cv2.normalize(absoul, None, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(f"normalized-{ctx.user.id}.png", normalize)
        await ctx.respond("Normalization applied!")
        f = hikari.File(f"normalized-{ctx.user.id}.png")
        await ctx.respond(f)
        time.sleep(1)
        os.remove(f"normalized-{ctx.user.id}.png")
    except:
        await ctx.respond("Error! Something went wrong." + traceback.format_exc())

@bot.command
@lightbulb.command("harris", "Not a filter, but this detects corners in the image.")
@lightbulb.implements(lightbulb.SlashCommand)
async def harris(ctx: lightbulb.context.Context):
    try:
        img = cv2.imread(f"originalimage-{ctx.user.id}.png")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray, 2, 3, 0.04)
        dst = cv2.dilate(dst, None)
        img[dst > 0.01 * dst.max()] = [0, 0, 255]
        cv2.imwrite(f"harris-{ctx.user.id}.png", img)
        await ctx.respond("Corners found!")
        f = hikari.File(f"harris-{ctx.user.id}.png")
        await ctx.respond(f)
        time.sleep(1)
        os.remove(f"harris-{ctx.user.id}.png")
    except:
        await ctx.respond("Error! Something went wrong." + traceback.format_exc())

@bot.command
@lightbulb.command("sobel", "Applies a sobel filter to the image.")
@lightbulb.implements(lightbulb.SlashCommand)
async def sobel(ctx: lightbulb.context.Context):
    try:
        img = cv2.imread(f"originalimage-{ctx.user.id}.png")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
        img = cv2.convertScaleAbs(img)
        cv2.imwrite(f"sobel-{ctx.user.id}.png", img)
        await ctx.respond("Sobel applied!")
        f = hikari.File(f"sobel-{ctx.user.id}.png")
        await ctx.respond(f)
        time.sleep(1)
        os.remove(f"contrast-{ctx.user.id}.png")
    except:
        await ctx.respond("Error! Something went wrong." + traceback.format_exc())

@bot.command
@lightbulb.command("canny", "Applies a canny filter to the image.")
@lightbulb.implements(lightbulb.SlashCommand)
async def canny(ctx: lightbulb.context.Context):
    try:
        img = cv2.imread(f"originalimage-{ctx.user.id}.png")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(gray, 100, 200)
        invert = cv2.bitwise_not(canny)
        cv2.imwrite(f"canny-{ctx.user.id}.png", invert)
        await ctx.respond("Canny applied!")
        f = hikari.File(f"canny-{ctx.user.id}.png")
        await ctx.respond(f)
        time.sleep(1)
        os.remove(f"canny-{ctx.user.id}.png")
    except:
        await ctx.respond("Error! Something went wrong." + traceback.format_exc())

@bot.command
@lightbulb.command("highpass", "Adds highpass to the image.")
@lightbulb.implements(lightbulb.SlashCommand)
async def highpass(ctx: lightbulb.context.Context):
    try:
        img = cv2.imread(f"originalimage-{ctx.user.id}.png")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (5, 5), 0), -4, 128)
        cv2.imwrite(f"highpass-{ctx.user.id}.png", img)
        await ctx.respond("Highpass applied!")
        f = hikari.File(f"highpass-{ctx.user.id}.png")
        await ctx.respond(f)
        time.sleep(1)
        os.remove(f"highpass-{ctx.user.id}.png")
    except:
        await ctx.respond("Error! Something went wrong." + traceback.format_exc())

@bot.command
@lightbulb.command("laplace", "Saturates the image.")
@lightbulb.implements(lightbulb.SlashCommand)
async def laplace(ctx: lightbulb.context.Context):
    try:
        img = cv2.imread(f"originalimage-{ctx.user.id}.png")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.Laplacian(img, cv2.CV_64F)
        img = cv2.convertScaleAbs(img)
        cv2.imwrite(f"laplace-{ctx.user.id}.png", img)
        await ctx.respond("Laplace applied!")
        f = hikari.File(f"laplace-{ctx.user.id}.png")
        await ctx.respond(f)
        time.sleep(1)
        os.remove(f"laplace-{ctx.user.id}.png")
    except:
        await ctx.respond("Error! Something went wrong." + traceback.format_exc())

@bot.command
@lightbulb.command("equalization", "Equalizes the image.")
@lightbulb.implements(lightbulb.SlashCommand)
async def equalization(ctx: lightbulb.context.Context):
    try:
        img = cv2.imread(f"originalimage-{ctx.user.id}.png")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        equ = cv2.equalizeHist(img)
        cv2.imwrite(f"equalization-{ctx.user.id}.png", equ)
        await ctx.respond("Equalization applied!")
        f = hikari.File(f"equalization-{ctx.user.id}.png")
        await ctx.respond(f)
        time.sleep(1)
        os.remove(f"equalization-{ctx.user.id}.png")
    except:
        await ctx.respond("Error! Something went wrong." + traceback.format_exc())

@bot.command
@lightbulb.command("skeleton", "Turns the image into a skeleton.")
@lightbulb.implements(lightbulb.SlashCommand)
async def skeleton(ctx: lightbulb.context.Context):
    try:
        img = cv2.imread(f"originalimage-{ctx.user.id}.png")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
        img = cv2.convertScaleAbs(img)
        img = cv2.dilate(img, None, iterations=2)
        img = cv2.erode(img, None, iterations=2)
        cv2.imwrite(f"skeleton-{ctx.user.id}.png", img)
        await ctx.respond("Skeleton applied!")
        f = hikari.File(f"skeleton-{ctx.user.id}.png")
        await ctx.respond(f)
        time.sleep(1)
        os.remove(f"skeleton-{ctx.user.id}.png")
    except:
        await ctx.respond("Error! Something went wrong." + traceback.format_exc())

@bot.command
@lightbulb.command("thinning", "Thins the lines of the image.")
@lightbulb.implements(lightbulb.SlashCommand)
async def thinning(ctx: lightbulb.context.Context):
    try:
        img = cv2.imread(f"originalimage-{ctx.user.id}.png")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        cv2.imwrite(f"thin-{ctx.user.id}.png", img)
        await ctx.respond("Lines thinned!")
        f = hikari.File(f"thin-{ctx.user.id}.png")
        await ctx.respond(f)
        time.sleep(1)
        os.remove(f"thin-{ctx.user.id}.png")
    except:
        await ctx.respond("Error! Something went wrong." + traceback.format_exc())

@bot.command
@lightbulb.command("thicken", "Thickens the lines of the image.")
@lightbulb.implements(lightbulb.SlashCommand)
async def thicken(ctx: lightbulb.context.Context):
    try:
        img = cv2.imread(f"originalimage-{ctx.user.id}.png")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        img = cv2.dilate(img, None, iterations=2)
        img = cv2.erode(img, None, iterations=2)
        cv2.imwrite(f"thicken-{ctx.user.id}.png", img)
        await ctx.respond("Lines thickened!")
        f = hikari.File(f"thicken-{ctx.user.id}.png")
        await ctx.respond(f)
        time.sleep(1)
        os.remove(f"thicken-{ctx.user.id}.png")
    except:
        await ctx.respond("Error! Something went wrong." + traceback.format_exc())

@bot.command
@lightbulb.command("line_filter", "Does what it says on the tin.")
@lightbulb.implements(lightbulb.SlashCommand)
async def linefilter(ctx: lightbulb.context.Context):
    try:
        img = cv2.imread(f"originalimage-{ctx.user.id}.png")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        cv2.imwrite(f"linefilter-{ctx.user.id}.png", img)
        await ctx.respond("Done!")
        f = hikari.File(f"linefilter-{ctx.user.id}.png")
        await ctx.respond(f)
        time.sleep(1)
        os.remove(f"linefilter-{ctx.user.id}.png")
    except:
        await ctx.respond("Error! Something went wrong." + traceback.format_exc())

@bot.command
@lightbulb.command("dft", "Converts image into DFT.")
@lightbulb.implements(lightbulb.SlashCommand)
async def dft(ctx: lightbulb.context.Context):
    try:
        img = cv2.imread(f"originalimage-{ctx.user.id}.png")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
        dftshift = np.fft.fftshift(dft)
        magnitudespectrum = 20 * np.log(cv2.magnitude(dftshift[:, :, 0], dftshift[:, :, 1]))
        cv2.imwrite(f"dft-{ctx.user.id}.png", magnitudespectrum)
        await ctx.respond("DFT applied!")
        f = hikari.File(f"dft-{ctx.user.id}.png")
        await ctx.respond(f)
        time.sleep(1)
        os.remove(f"dft-{ctx.user.id}.png")
    except:
        await ctx.respond("Error! Something went wrong." + traceback.format_exc())

@bot.command
@lightbulb.command("dst", "Does the same as DFT.")
@lightbulb.implements(lightbulb.SlashCommand)
async def dst(ctx: lightbulb.context.Context):
    try:
        img = cv2.imread(f"originalimage-{ctx.user.id}.png")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dst = cv2.dilate(img, None, iterations=2)
        dst = cv2.erode(dst, None, iterations=2)
        cv2.imwrite(f"dst-{ctx.user.id}.png", dst)
        await ctx.respond("DST applied!")
        f = hikari.File(f"dst-{ctx.user.id}.png")
        await ctx.respond(f)
        time.sleep(1)
        os.remove(f"dst-{ctx.user.id}.png")
    except:
        await ctx.respond("Error! Something went wrong." + traceback.format_exc())

@bot.command
@lightbulb.command("mirror", "Mirrors the image.")
@lightbulb.implements(lightbulb.SlashCommand)
async def mirror(ctx: lightbulb.context.Context):
    try:
        img = cv2.imread(f"originalimage-{ctx.user.id}.png")
        flip = cv2.flip(img, 1)
        cv2.imwrite(f"mirror-{ctx.user.id}.png", flip)
        await ctx.respond("Mirror applied!")
        f = hikari.File(f"mirror-{ctx.user.id}.png")
        await ctx.respond(f)
        time.sleep(1)
        os.remove(f"mirror-{ctx.user.id}.png")
    except:
        await ctx.respond("Error! Something went wrong." + traceback.format_exc())

@bot.command
@lightbulb.command("otsu", "Applies otsu to the image.")
@lightbulb.implements(lightbulb.SlashCommand)
async def otsu(ctx: lightbulb.context.Context):
    try:
        img = cv2.imread(f"originalimage-{ctx.user.id}.png")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        cv2.imwrite(f"otsu-{ctx.user.id}.png", thresh)
        await ctx.respond("Otsu applied!")
        f = hikari.File(f"otsu-{ctx.user.id}.png")
        await ctx.respond(f)
        time.sleep(1)
        os.remove(f"otsu-{ctx.user.id}.png")
    except:
        await ctx.respond("Error! Something went wrong." + traceback.format_exc())

@bot.command
@lightbulb.command("upscale", "Upscales the image (Not AI).")
@lightbulb.implements(lightbulb.SlashCommand)
async def upscale(ctx: lightbulb.context.Context):
    try:
        img = cv2.imread(f"originalimage-{ctx.user.id}.png")
        resized = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(f"upscaled-{ctx.user.id}.png", resized)
        await ctx.respond("Upscaled!")
        f = hikari.File(f"upscaled-{ctx.user.id}.png")
        await ctx.respond(f)
        time.sleep(1)
        os.remove(f"upscaled-{ctx.user.id}.png")
    except:
        await ctx.respond("Error! Something went wrong." + traceback.format_exc())

@bot.command
@lightbulb.command("downscale", "Downscales the image (Not AI).")
@lightbulb.implements(lightbulb.SlashCommand)
async def downscale(ctx: lightbulb.context.Context):
    try:
        img = cv2.imread(f"originalimage-{ctx.user.id}.png")
        resized = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(f"downscaled-{ctx.user.id}.png", resized)
        await ctx.respond("Downscaled!")
        f = hikari.File(f"downscaled-{ctx.user.id}.png")
        await ctx.respond(f)
        time.sleep(1)
        os.remove(f"downscaled-{ctx.user.id}.png")
    except:
        await ctx.respond("Error! Something went wrong." + traceback.format_exc())

@bot.command
@lightbulb.command("noise", "Adds noise to the image.")
@lightbulb.implements(lightbulb.SlashCommand)
async def noise(ctx: lightbulb.context.Context):
    try:
        img = cv2.imread(f"originalimage-{ctx.user.id}.png")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        noise = np.random.randint(0, 255, gray.shape)
        noise = noise.astype(np.uint8)
        noise = cv2.add(gray, noise)
        cv2.imwrite(f"noise-{ctx.user.id}.png", noise)
        await ctx.respond("Noise applied!")
        f = hikari.File(f"noise-{ctx.user.id}.png")
        await ctx.respond(f)
        time.sleep(1)
        os.remove(f"noise-{ctx.user.id}.png")
    except:
        await ctx.respond("Error! Something went wrong." + traceback.format_exc())

@bot.command
@lightbulb.command("highboost", "Highboosts the image.")
@lightbulb.implements(lightbulb.SlashCommand)
async def highboost(ctx: lightbulb.context.Context):
    try:
        img = cv2.imread(f"originalimage-{ctx.user.id}.png")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dst = cv2.Laplacian(gray, cv2.CV_64F)
        dst = np.uint8(np.absolute(dst))
        cv2.imwrite(f"highboost-{ctx.user.id}.png", dst)
        await ctx.respond("Highboost applied!")
        f = hikari.File(f"highboost-{ctx.user.id}.png")
        await ctx.respond(f)
        time.sleep(1)
        os.remove(f"highboost-{ctx.user.id}.png")
    except:
        await ctx.respond("Error! Something went wrong." + traceback.format_exc())

@bot.command
@lightbulb.command("tophat", "Applies a tophat filter (not an actual top hat).")
@lightbulb.implements(lightbulb.SlashCommand)
async def tophat(ctx: lightbulb.context.Context):
    try:
        img = cv2.imread(f"originalimage-{ctx.user.id}.png")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((5, 5), np.uint8)
        opening = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        cv2.imwrite(f"tophat-{ctx.user.id}.png", opening)
        await ctx.respond("Tophat applied!")
        f = hikari.File(f"tophat-{ctx.user.id}.png")
        await ctx.respond(f)
        time.sleep(1)
        os.remove(f"tophat-{ctx.user.id}.png")
    except:
        await ctx.respond("Error! Something went wrong." + traceback.format_exc())

@bot.command
@lightbulb.command("blackhat", "Applies a blackhat filter.")
@lightbulb.implements(lightbulb.SlashCommand)
async def blackhat(ctx: lightbulb.context.Context):
    try:
        img = cv2.imread(f"originalimage-{ctx.user.id}.png")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((5, 5), np.uint8)
        closing = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        cv2.imwrite(f"blackhat-{ctx.user.id}.png", closing)
        await ctx.respond("Blackhat applied!")
        f = hikari.File(f"blackhat-{ctx.user.id}.png")
        await ctx.respond(f)
        time.sleep(1)
        os.remove(f"blackhat-{ctx.user.id}.png")
    except:
        await ctx.respond("Error! Something went wrong." + traceback.format_exc())

@bot.command
@lightbulb.option("power", "Amount.", type=int, required=True)
@lightbulb.command("powerlaw", "Not sure what this one does.")
@lightbulb.implements(lightbulb.SlashCommand)
async def powerlaw(ctx: lightbulb.context.Context):
    try:
        img = cv2.imread(f"originalimage-{ctx.user.id}.png")
        power = ctx.options.power
        img = cv2.pow(img, power)
        cv2.imwrite(f"powerlaw-{ctx.user.id}.png", img)
        await ctx.respond("Powerlaw applied!")
        f = hikari.File(f"powerlaw-{ctx.user.id}.png")
        await ctx.respond(f)
        time.sleep(1)
        os.remove(f"powerlaw-{ctx.user.id}.png")
    except:
        await ctx.respond("Error! Something went wrong." + traceback.format_exc())

@bot.command
@lightbulb.command("greenchannel", "Makes the image only green.")
@lightbulb.implements(lightbulb.SlashCommand)
async def greenchannel(ctx: lightbulb.context.Context):
    try:
        img = cv2.imread(f"originalimage-{ctx.user.id}.png")
        img[:,:,0] = 0
        img[:,:,2] = 0
        cv2.imwrite(f"green-{ctx.user.id}.png", img)
        await ctx.respond("Green channel applied!")
        f = hikari.File(f"contgreenrast-{ctx.user.id}.png")
        await ctx.respond(f)
        time.sleep(1)
        os.remove(f"green-{ctx.user.id}.png")
    except:
        await ctx.respond("Error! Something went wrong." + traceback.format_exc())

@bot.command
@lightbulb.command("yiq", "Converts image to YIQ.")
@lightbulb.implements(lightbulb.SlashCommand)
async def yiq(ctx: lightbulb.context.Context):
    try:
        img = cv2.imread(f"originalimage-{ctx.user.id}.png")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        img[:,:,0] = img[:,:,0] * 1.5
        img[:,:,1] = img[:,:,1] * 1.5
        img[:,:,2] = img[:,:,2] * 1.5
        cv2.imwrite(f"yiq-{ctx.user.id}.png", img)
        await ctx.respond("Converted to YIQ!")
        f = hikari.File(f"yiq-{ctx.user.id}.png")
        await ctx.respond(f)
        time.sleep(1)
        os.remove(f"yiq-{ctx.user.id}.png")
    except:
        await ctx.respond("Error! Something went wrong." + traceback.format_exc())

@bot.command
@lightbulb.command("yuv", "Converts the image to YUV.")
@lightbulb.implements(lightbulb.SlashCommand)
async def yuv(ctx: lightbulb.context.Context):
    try:
        img = cv2.imread(f"originalimage-{ctx.user.id}.png")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img[:,:,0] = img[:,:,0] * 1.5
        img[:,:,1] = img[:,:,1] * 1.5
        img[:,:,2] = img[:,:,2] * 1.5
        cv2.imwrite(f"yuv-{ctx.user.id}.png", img)
        await ctx.respond("Converted to YUV!")
        f = hikari.File(f"yuv-{ctx.user.id}.png")
        await ctx.respond(f)
        time.sleep(1)
        os.remove(f"yuv-{ctx.user.id}.png")
    except:
        await ctx.respond("Error! Something went wrong." + traceback.format_exc())

@bot.command
@lightbulb.command("hsl", "Converts image into HSL.")
@lightbulb.implements(lightbulb.SlashCommand)
async def hsl(ctx: lightbulb.context.Context):
    try:
        img = cv2.imread(f"originalimage-{ctx.user.id}.png")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img[:,:,0] = img[:,:,0] * 1.5
        img[:,:,1] = img[:,:,1] * 1.5
        img[:,:,2] = img[:,:,2] * 1.5
        cv2.imwrite(f"hsl-{ctx.user.id}.png", )
        await ctx.respond("Converted to HSL!")
        f = hikari.File(f"hsl-{ctx.user.id}.png")
        await ctx.respond(f)
        time.sleep(1)
        os.remove(f"hsl-{ctx.user.id}.png")
    except:
        await ctx.respond("Error! Something went wrong." + traceback.format_exc())

@bot.command
@lightbulb.command("hls", "Converts image to HLS.")
@lightbulb.implements(lightbulb.SlashCommand)
async def hls(ctx: lightbulb.context.Context):
    try:
        img = cv2.imread(f"originalimage-{ctx.user.id}.png")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        img[:,:,0] = img[:,:,0] * 1.5
        img[:,:,1] = img[:,:,1] * 1.5
        img[:,:,2] = img[:,:,2] * 1.5
        cv2.imwrite(f"hls-{ctx.user.id}.png", img)
        await ctx.respond("HLS applied!")
        f = hikari.File(f"hls-{ctx.user.id}.png")
        await ctx.respond(f)
        time.sleep(1)
        os.remove(f"hls-{ctx.user.id}.png")
    except:
        await ctx.respond("Error! Something went wrong." + traceback.format_exc())

@bot.command
@lightbulb.command("cie", "Converts image to CIE.")
@lightbulb.implements(lightbulb.SlashCommand)
async def cie(ctx: lightbulb.context.Context):
    try:
        img = cv2.imread(f"originalimage-{ctx.user.id}.png")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2XYZ)
        img[:,:,0] = img[:,:,0] * 1.5
        img[:,:,1] = img[:,:,1] * 1.5
        img[:,:,2] = img[:,:,2] * 1.5
        cv2.imwrite(f"cie-{ctx.user.id}.png", img)
        await ctx.respond("CIE applied!")
        f = hikari.File(f"cie-{ctx.user.id}.png")
        await ctx.respond(f)
        time.sleep(1)
        os.remove(f"cie-{ctx.user.id}.png")
    except:
        await ctx.respond("Error! Something went wrong." + traceback.format_exc())

@bot.command
@lightbulb.command("corrupt", "Corrupts the image.")
@lightbulb.implements(lightbulb.SlashCommand)
async def corrupt(ctx: lightbulb.context.Context):
    try:
        img = cv2.imread(f"originalimage-{ctx.user.id}.png")
        height, width, channels = img.shape
        for x in range(0, width):
            for y in range(0, height):
                img[y, x] = (random.randint(0, 200), random.randint(0, 200), random.randint(0, 200))
        cv2.imwrite(f"corrupted-{ctx.user.id}.png", img)
        await ctx.respond("Corruption applied!")
        f = hikari.File(f"corrupted-{ctx.user.id}.png")
        await ctx.respond(f)
        time.sleep(1)
        os.remove(f"corrupted-{ctx.user.id}.png")
    except:
        await ctx.respond("Error! Something went wrong." + traceback.format_exc())

@bot.command
@lightbulb.command("stippling", "Adds a stippling effect to the image.")
@lightbulb.implements(lightbulb.SlashCommand)
async def stippling(ctx: lightbulb.context.Context):
    try:
        img = cv2.imread(f"originalimage-{ctx.user.id}.png")
        height, width, channels = img.shape
        for x in range(0, width):
            for y in range(0, height):
                if x % 2 == 0:
                    img[y, x] = (random.randint(0, 200), random.randint(0, 200), random.randint(0, 200))
        cv2.imwrite(f"stippling-{ctx.user.id}.png", )
        await ctx.respond("Stippling applied!")
        f = hikari.File(f"stippling-{ctx.user.id}.png")
        await ctx.respond(f)
        time.sleep(1)
        os.remove(f"stippling-{ctx.user.id}.png")
    except:
        await ctx.respond("Error! Something went wrong." + traceback.format_exc())

@bot.command
@lightbulb.command("contraststretch", "Self explanatory.")
@lightbulb.implements(lightbulb.SlashCommand)
async def contraststretch(ctx: lightbulb.context.Context):
    try:
        img = cv2.imread(f"originalimage-{ctx.user.id}.png")
        h, w, c = img.shape
        for x in range(0, w):
            for y in range(0, h):
                img[y, x] = (int(img[y, x][0] * 1.5), int(img[y, x][1] * 1.5), int(img[y, x][2] * 1.5))
        cv2.imwrite(f"contraststretch-{ctx.user.id}.png", img)
        await ctx.respond("Filter applied!")
        f = hikari.File(f"contraststretch-{ctx.user.id}.png")
        await ctx.respond(f)
        time.sleep(1)
        os.remove(f"contraststretch-{ctx.user.id}.png")
    except:
        await ctx.respond("Error! Something went wrong." + traceback.format_exc())

@bot.command
@lightbulb.command("quantize", "Quantizes the image.")
@lightbulb.implements(lightbulb.SlashCommand)
async def quantize(ctx: lightbulb.context.Context):
    try:
        img = cv2.imread(f"originalimage-{ctx.user.id}.png")
        h, w, c = img.shape
        for x in range(0, w):
            for y in range(0, h):
                img[y, x] = (int(img[y, x][0] / 2) * 2, int(img[y, x][1] / 2) * 2, int(img[y, x][2] / 2) * 2)
        cv2.imwrite(f"quantize-{ctx.user.id}.png", img)
        await ctx.respond("Quantize applied!")
        f = hikari.File(f"quantize-{ctx.user.id}.png")
        await ctx.respond(f)
        time.sleep(1)
        os.remove(f"quantize-{ctx.user.id}.png")
    except:
        await ctx.respond("Error! Something went wrong." + traceback.format_exc())

@bot.command
@lightbulb.command("acidtrip", "Whoa man those are some mighty shrooms.")
@lightbulb.implements(lightbulb.SlashCommand)
async def acidtrip(ctx: lightbulb.context.Context):
    try:
        img = cv2.imread(f"originalimage-{ctx.user.id}.png")
        h, w, c = img.shape
        for x in range(0, w):
            for y in range(0, h):
                img[y, x] = (int(math.pow(img[y, x][0], 1.5)), int(math.pow(img[y, x][1], 1.5)), int(math.pow(img[y, x][2], 1.5)))
        cv2.imwrite(f"acidtrip-{ctx.user.id}.png", img)
        await ctx.respond("Acid trip applied!")
        f = hikari.File(f"acidtrip-{ctx.user.id}.png")
        await ctx.respond(f)
        time.sleep(1)
        os.remove(f"acidtrip-{ctx.user.id}.png")
    except:
        await ctx.respond("Error! Something went wrong." + traceback.format_exc())

@bot.command
@lightbulb.command("bluechannel", "Filters by blue.")
@lightbulb.implements(lightbulb.SlashCommand)
async def bluechannel(ctx: lightbulb.context.Context):
    try:
        img = cv2.imread(f"originalimage-{ctx.user.id}.png")
        h, w, c = img.shape
        for x in range(0, w):
            for y in range(0, h):
                img[y, x] = (img[y, x][0], 0, 0)
        cv2.imwrite(f"bluechannel-{ctx.user.id}.png", img)
        await ctx.respond("Blue colors filtered!")
        f = hikari.File(f"bluechannel-{ctx.user.id}.png")
        await ctx.respond(f)
        time.sleep(1)
        os.remove(f"bluechannel-{ctx.user.id}.png")
    except:
        await ctx.respond("Error! Something went wrong." + traceback.format_exc())

@bot.command
@lightbulb.command("redchannel", "Filters by red.")
@lightbulb.implements(lightbulb.SlashCommand)
async def redchannel(ctx: lightbulb.context.Context):
    try:
        img = cv2.imread(f"originalimage-{ctx.user.id}.png")
        h, w, c = img.shape
        for x in range(0, w):
            for y in range(0, h):
                img[y, x] = (0, 0, img[y, x][2])
        cv2.imwrite(f"redchannel-{ctx.user.id}.png", )
        await ctx.respond("Red colors filtered!")
        f = hikari.File(f"redchannel-{ctx.user.id}.png")
        await ctx.respond(f)
        time.sleep(1)
        os.remove(f"redchannel-{ctx.user.id}.png")
    except:
        await ctx.respond("Error! Something went wrong." + traceback.format_exc())

@bot.command
@lightbulb.command("cyanchannel", "Filters the image by cyan.")
@lightbulb.implements(lightbulb.SlashCommand)
async def cyanchannel(ctx: lightbulb.context.Context):
    try:
        img = cv2.imread(f"originalimage-{ctx.user.id}.png")
        h, w, c = img.shape
        for x in range(0, w):
            for y in range(0, h):
                img[y, x] = (img[y, x][0], img[y, x][2], 0)
        cv2.imwrite(f"cyan-{ctx.user.id}.png", img)
        await ctx.respond("Cyan filtered!")
        f = hikari.File(f"cyan-{ctx.user.id}.png")
        await ctx.respond(f)
        time.sleep(1)
        os.remove(f"cyan-{ctx.user.id}.png")
    except:
        await ctx.respond("Error! Something went wrong." + traceback.format_exc())

@bot.command
@lightbulb.command("purplechannel", "Filters the image by purple.")
@lightbulb.implements(lightbulb.SlashCommand)
async def purplechannel(ctx: lightbulb.context.Context):
    try:
        img = cv2.imread(f"originalimage-{ctx.user.id}.png")
        h, w, c = img.shape
        for x in range(0, w):
            for y in range(0, h):
                img[y, x] = (img[y, x][1], 0, img[y, x][2])
        cv2.imwrite(f"purple-{ctx.user.id}.png", img)
        await ctx.respond("Purple filter applied!")
        f = hikari.File(f"purple-{ctx.user.id}.png")
        await ctx.respond(f)
        time.sleep(1)
        os.remove(f"purple-{ctx.user.id}.png")
    except:
        await ctx.respond("Error! Something went wrong." + traceback.format_exc())

@bot.command
@lightbulb.command("scharr", "Applies scharr filter.")
@lightbulb.implements(lightbulb.SlashCommand)
async def scharr(ctx: lightbulb.context.Context):
    try:
        img = cv2.imread(f"originalimage-{ctx.user.id}.png")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.Scharr(img, cv2.CV_8U, 1, 0)
        cv2.imwrite(f"scharr-{ctx.user.id}.png", img)
        await ctx.respond("Scharr applied!")
        f = hikari.File(f"scharr-{ctx.user.id}.png")
        await ctx.respond(f)
        time.sleep(1)
        os.remove(f"scharr-{ctx.user.id}.png")
    except:
        await ctx.respond("Error! Something went wrong." + traceback.format_exc())

@bot.command
@lightbulb.command("bubbles", "Applies bubbles filter.")
@lightbulb.implements(lightbulb.SlashCommand)
async def bubbles(ctx: lightbulb.context.Context):
    try:
        img = cv2.imread(f"originalimage-{ctx.user.id}.png")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.bilateralFilter(img, 9, 75, 75)
        cv2.imwrite(f"bubbles-{ctx.user.id}.png", img)
        await ctx.respond("Bubbles applied!")
        f = hikari.File(f"bubbles-{ctx.user.id}.png")
        await ctx.respond(f)
        time.sleep(1)
        os.remove(f"bubbles-{ctx.user.id}.png")
    except:
        await ctx.respond("Error! Something went wrong." + traceback.format_exc())

@bot.command
@lightbulb.command("hough", "Uses HoughCircles.")
@lightbulb.implements(lightbulb.SlashCommand)
async def hough(ctx: lightbulb.context.Context):
    try:
        img = cv2.imread(f"originalimage-{ctx.user.id}.png")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 60, param1=50, param2=30, minRadius=0, maxRadius=0)
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
        cv2.imwrite(f"hough-{ctx.user.id}.png", )
        await ctx.respond("Hough circles applied!")
        f = hikari.File(f"hough-{ctx.user.id}.png")
        await ctx.respond(f)
        time.sleep(1)
        os.remove(f"hough-{ctx.user.id}.png")
    except:
        await ctx.respond("Error! Something went wrong." + traceback.format_exc())

@bot.command
@lightbulb.command("mandelbrot", "Applies mandelbrot")
@lightbulb.implements(lightbulb.SlashCommand)
async def mandelbrot(ctx: lightbulb.context.Context):
    try:
        img = cv2.imread(f"originalimage-{ctx.user.id}.png")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        filter = cv2.bilateralFilter(gray, 9, 75, 75)
        blur = cv2.medianBlur(filter, 5)
        blur2 = cv2.GaussianBlur(blur, (5, 5), 0)
        canny = cv2.Canny(blur2, 100, 200)
        cv2.imwrite(f"mandelbrot-{ctx.user.id}.png", )
        await ctx.respond("Mandelbrot applied!")
        f = hikari.File(f"mandelbrot-{ctx.user.id}.png")
        await ctx.respond(f)
        time.sleep(1)
        os.remove(f"mandelbrot-{ctx.user.id}.png")
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
