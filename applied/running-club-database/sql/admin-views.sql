CREATE VIEW Admin_basic_info
AS SELECT
	adminname AS '姓名',
    adminid AS '管理员ID',
    adminposition AS '职务'
FROM Admin;

CREATE VIEW Admin_member_act
AS SELECT
	COUNT(DISTINCT m.mid) AS '会员人数',
    SUM(ma.mactmile) AS '会员本年度跑步总里程',
    AVG(weekly.total_mile) AS '会员本周跑步平均里程'
FROM Member m
LEFT JOIN MAct ma ON (m.mid = ma.mid
AND YEAR(ma.macttime) = YEAR(CURRENT_DATE))
LEFT JOIN
	(SELECT 
		mid,
		SUM(mactmile) AS total_mile
	FROM MAct
	WHERE WEEK(macttime) = WEEK(CURRENT_DATE)
	AND YEAR(macttime) = YEAR(CURRENT_DATE)
	GROUP BY mid) weekly ON m.mid = weekly.mid;
    
CREATE VIEW Sponsor_info
AS SELECT
sp.sponsorid AS '赞助商ID',
sp.sponsorname AS '赞助商名称',
sp.sponsortime AS '合作开始时间',
sp.sponsorstatus AS '合作状态'
FROM Sponsor sp;

CREATE VIEW Train_review
AS SELECT
	t.trainid AS '训练计划ID',
    t.traincontent AS '训练内容',
    tgs.tgstatus AS '审核申请状态'
FROM Train t
LEFT JOIN TGSta tgs ON tgs.trainid=t.trainid
GROUP BY t.trainid,t.traincontent;

CREATE VIEW Contest_review
AS SELECT
	c.contestid AS '竞赛ID',
    c.contestname AS '竞赛名称',
    cs.conteststatus AS '审核情况'
FROM Contest c
LEFT JOIN CSta cs ON cs.contestid = c.contestid
GROUP BY c.contestid,c.contestname;




