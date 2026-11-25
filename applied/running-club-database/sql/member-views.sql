CREATE VIEW User_basic_info
AS SELECT  
	mname AS '姓名',
    studentid AS '学号',
    msex AS '性别',
    mage AS '年龄',
    mcampus AS '所属校区',
    mtel AS '联系方式',
    menrolltime AS '注册时间'
FROM Member;

CREATE VIEW User_health_info
AS SELECT 
	mp.mphyh AS '身高',
    mp.mphyw AS '体重',
	ROUND(mp.mphyw/POWER(mp.mphyh,2),2) AS 'BMI',
	mp.mphybfp AS '体脂率',
    AVG(ma.macthr) AS '平均心率',
	SUM(ma.mactmile) AS "跑步总里程",
	mp.mphydisease AS '相关病史'
FROM Member m
LEFT JOIN MPhy mp ON m.mid=mp.mid
LEFT JOIN MAct ma ON m.mid=ma.mid
GROUP BY m.mid;

CREATE VIEW Weekly_mile_rank
AS SELECT 
	RANK() OVER (ORDER BY total_mile DESC) AS 'weekrank',
	mid AS '用户ID',
    total_mile AS '本周运动总里程'
	FROM 
		(SELECT 
			mid,
			SUM(mactmile) AS 'total_mile'
			FROM MAct
			WHERE WEEK(macttime)=WEEK(CURRENT_DATE)
			AND YEAR(macttime)=YEAR(CURRENT_DATE)
			GROUP BY mid) t
ORDER BY weekrank;

CREATE VIEW Train_registration AS 
SELECT
	t.trainid AS '训练计划ID',
    t.traincontent AS '训练内容',
    t.traintime AS '训练时间',
	COUNT(TE.mid) AS '已报名人数'
FROM Train t
LEFT JOIN TE ON t.trainid=TE.trainid
WHERE t.traintime > CURRENT_DATE
GROUP BY t.trainid, t.traincontent, t.traintime;

 
CREATE VIEW Contest_registration AS 
SELECT
	c.contestname AS '竞赛名称',
    c.contesttime AS '竞赛时间',
    c.contestprize AS '竞赛奖项',
	COUNT(CE.mid) AS '已报名人数'
FROM Contest c
LEFT JOIN CE ON c.contestid = CE.contestid
WHERE c.contesttime > CURRENT_DATE
GROUP BY c.contestname, c.contesttime, c.contestprize;


CREATE VIEW Item_Distribution
AS SELECT
	i.itemname AS '物资名称',
    MI.itemquant AS '物资数量',
    c.contesttime AS '竞赛时间'
FROM Item i
LEFT JOIN MI ON MI.itemid =i.itemid
LEFT JOIN CE ON MI.mid=CE.mid
LEFT JOIN Contest c ON CE.contestid = c.contestid
WHERE MI.mistatus = '未领取';

CREATE VIEW Fitness_report
AS SELECT
	r.reportid AS '报告ID',
    r.actfrequency AS '运动频率',
    r.acttype AS '运动类型分布',
    r.actmile AS '平均里程',
    r.acthr AS '平均心率',
    r.bfp AS '体脂率',
    r.advice AS '建议'
FROM Report r
JOIN Member m ON r.mid=m.mid
WHERE reporttime=CURRENT_DATE;

    




