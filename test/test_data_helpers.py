
import unittest

from detect.util.data_helpers import DataHelper
from newspaper import Article


class TestDataHelpers(unittest.TestCase):
    
    def test2(self):
        articles =[
         'http://www.radionz.co.nz/news/national/281869/seven-arrests-over-police-manhunt',
         'http://m.nzherald.co.nz/wanganui-chronicle/news/article.cfm?c_id=1503426&objectid=11491573',
         'http://m.nzherald.co.nz/wanganui-chronicle/news/article.cfm?c_id=1503426&objectid=11580358',
         'http://m.nzherald.co.nz/wanganui-chronicle/news/article.cfm?c_id=1503426&objectid=11580350',
         'http://www.stuff.co.nz/national/crime/75978990/whanganui-woman-accused-of-leaving-child-in-car-overnight.html',
         'http://m.nzherald.co.nz/wanganui-chronicle/news/article.cfm?c_id=1503426&objectid=11574608',
         'http://m.nzherald.co.nz/wanganui-chronicle/news/article.cfm?c_id=1503426&objectid=11577923',
         'http://www.nzherald.co.nz/wanganui-chronicle/news/article.cfm?c_id=1503426&objectid=11591401',
         'http://m.nzherald.co.nz/wanganui-chronicle/news/article.cfm?c_id=1503426&objectid=11566180'
         ]
        
        articles = [
         'http://www.express.co.uk/news/uk/657926/New-Zealand-John-Key-slams-David-Cameron-Britain-forgetting-history-European-Union-EU',
         'http://www.bbc.co.uk/news/uk-wales-35954982',
         'http://www.telegraph.co.uk/news/2016/04/04/david-cameron-will-be-an-excellent-former-prime-minister/',
         'http://www.pressandjournal.co.uk/fp/news/aberdeenshire/880519/david-camerons-father-named-panamanian-company-aberdeenshire-home/',
         'http://www.theguardian.com/politics/2016/apr/01/senior-tories-brexit-vote-leave-attacks-david-cameron-letter-nhs-staff',
         'http://www.dailymail.co.uk/news/article-3519908/Nuclear-drones-threat-British-cities-Cameron-Obama-hold-war-game-session-respond-attack-kill-thousands-people.html',
         'http://www.telegraph.co.uk/news/2016/03/31/if-david-cameron-cant-stop-the-tory-fighting-hell-clear-jeremy-c/',
         'http://www.manchestereveningnews.co.uk/news/greater-manchester-news/gmp-boost-number-armed-officers-11125178',
         'http://www.theguardian.com/commentisfree/2016/apr/03/cameron-headphones-what-is-cool-what-is-not']
        
        with open("./Output2.txt", "w") as text_file:
            for url in articles:
                print(url)
                a = Article(url)
                a.download()
                a.parse()
                text_file.write(a.text.encode('utf-8'))
                text_file.write('\n')


    def test(self):
        training_data = [
            ([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], "data/profiles/train/davidcameron_1.txt"),
            ([0.0, 1.0, 0.0, 0.0, 0.0, 0.0], "data/profiles/train/davidcameron_2.txt"),
            ([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], "data/profiles/train/davidcameron_3.txt"),
            ([0.0, 0.0, 0.0, 1.0, 0.0, 0.0], "data/profiles/train/davidcameron_4.txt"),
            ([0.0, 0.0, 0.0, 0.0, 1.0, 0.0], "data/profiles/train/davidcameron_5.txt"),
            ([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], "data/profiles/train/davidcameron_6.txt")]

        # x_text, y = self.dataHelper.load_data_and_labels(training_data)

        data_helper = DataHelper()

        x, y, voc, vocInv = data_helper.load_data(training_data, 50)
        print(vocInv)

        batches = data_helper.batch_iter(x, y, 50, 1, shuffle=True)
        for batch_x, batch_y in batches:
            print(batch_y[5, :])
#             temp = [vocInv[i] for i in batch_x[5, :]]
            print(batch_x[5, :])