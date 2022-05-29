# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 16:51:33 2022

@author: dbvks
"""    
        
import pymysql
 
class Cctv:
    def __init__(self):
        pass
    
    def getc(self, id):
        ret = []
        db = pymysql.connect(host='localhost', user='root', db='motionguard', password='dkemysql244', charset='utf8')
        curs = db.cursor()
        
        sql = "select * from cctv where id=%s";
        curs.execute(sql,(id))
        
        row = curs.fetchone()
        
        for e in row:
            temp ={'id':e[0],'title':e[1],'contents':e[2],'state':e[3],'is_deleted':e[4],'created_at':e[5],'updated_at':e[6] }
            ret.append(temp)
        
        db.commit()
        db.close()
        return row
    
    def getCctv(self):
        ret = []
        db = pymysql.connect(host='localhost', user='root', db='motionguard', password='dkemysql244', charset='utf8')
        curs = db.cursor()
        
        sql = "select * from cctv";
        curs.execute(sql)
        
        rows = curs.fetchall()
        for e in rows:
            temp = {'id':e[0],'title':e[1],'contents':e[2],'state':e[3],'is_deleted':e[4],'created_at':e[5],'updated_at':e[6] }
            ret.append(temp)
        
        db.commit()
        db.close()
        return ret
    
    def insCctv(self, title, contents):
        db = pymysql.connect(host='localhost', user='root', db='motionguard', password='dkemysql244', charset='utf8')
        curs = db.cursor()
        
        sql = '''insert into cctv (title, contents) values(%s,%s)'''
        curs.execute(sql,(title, contents))
        db.commit()
        db.close()
    
    def updCctv(self, id, title, contents): 
        db = pymysql.connect(host='localhost', user='root', db='motionguard', password='dkemysql244', charset='utf8')
        curs = db.cursor()
        
        sql = "update user set title=%s, contents=%s where id=%s"
        curs.execute(sql,(title, contents, id))
        db.commit()
        db.close()
        
    def updState(self, id, state): 
        db = pymysql.connect(host='localhost', user='root', db='motionguard', password='dkemysql244', charset='utf8')
        curs = db.cursor()
        
        sql = "update user set state=%s where id=%s"
        curs.execute(sql,(state, id))
        db.commit()
        db.close()
        
    def delCctv(self, id):
        db = pymysql.connect(host='localhost', user='root', db='motionguard', password='dkemysql244', charset='utf8')
        curs = db.cursor()
        
        sql = "delete from cctv where id=%s"
        curs.execute(sql, id)
        db.commit()
        db.close()
 
if __name__ == '__main__':
    #MyEmpDao().insEmp('aaa', 'bb', 'cc', 'dd')
    #MyEmpDao().updEmp('aa', 'dd', 'dd', 'aa')
    #MyEmpDao().delEmp('aaa')
    cctvList = Cctv().getCctv();
    print(cctvList)