CREATE TABLE Member
(mid VARCHAR(20) PRIMARY KEY,
studentid VARCHAR(20) NOT NULL,
mname VARCHAR(20) NOT NULL,
msex VARCHAR(5) NOT NULL,
mage VARCHAR(5) NOT NULL,
mtel VARCHAR(20) NOT NULL,
mcampus VARCHAR(20),
menrolltime DATE NOT NULL,
mupdatetime DATE NOT NULL);

CREATE TABLE MPhy
(mid VARCHAR(20) NOT NULL,
mphytime VARCHAR(20) NOT NULL,
mphyh DECIMAL(4,3) NOT NULL,
mphyw DECIMAL(5,2) NOT NULL,
mphybfp DECIMAL(4,2),
mphydisease VARCHAR(100),
PRIMARY KEY (mid,mphytime),
FOREIGN KEY (mid) REFERENCES Member(mid));
  
CREATE TABLE MAct 
(mid VARCHAR(20) NOT NULL,
macttime DATE NOT NULL,
macttype VARCHAR(20),
mactmile DECIMAL(5,2) NOT NULL,
macthr INT NOT NULL,
PRIMARY KEY (mid, macttime),
FOREIGN KEY (mid) REFERENCES Member(mid));

CREATE TABLE Admin
(adminid VARCHAR(20) PRIMARY KEY,
adminname VARCHAR(20) NOT NULL,
adminposition VARCHAR(50) NOT NULL);

CREATE TABLE UpdateRecord (
  updateid VARCHAR(20) PRIMARY KEY,
  updatetime DATE NOT NULL,
  updateattribute VARCHAR(20) NOT NULL,
  updatecontent VARCHAR(100) NOT NULL,
  adminid VARCHAR(20) NOT NULL,   -- 先定义列
  mid VARCHAR(20) NOT NULL,       -- 再定义列
  FOREIGN KEY (adminid) REFERENCES Admin(adminid),
  FOREIGN KEY (mid) REFERENCES Member(mid)
);

CREATE TABLE USta 
(usid VARCHAR(20) PRIMARY KEY,
 updateid VARCHAR(20) NOT NULL,
 auditstatus VARCHAR(20) NOT NULL,
 FOREIGN KEY (updateid) REFERENCES UpdateRecord(updateid));

CREATE TABLE Gym 
(gymid VARCHAR(20) PRIMARY KEY,
gymname VARCHAR(20) NOT NULL,
gymtype VARCHAR(20) NOT NULL,
gymvolume INT NOT NULL,
gymlocation VARCHAR(50) NOT NULL);

CREATE TABLE Train
(trainid VARCHAR(20) PRIMARY KEY,
 traincontent VARCHAR(50) NOT NULL,
 traintime DATE NOT NULL);

CREATE TABLE TSta
(tsid VARCHAR(20) PRIMARY KEY,
 trainid VARCHAR(20) NOT NULL,
 trainstatus VARCHAR(20) NOT NULL,
 FOREIGN KEY (trainid) REFERENCES Train(trainid));

CREATE TABLE TG 
(trainid VARCHAR(20) NOT NULL,
 gymid VARCHAR(20) NOT NULL,
 tgtime DATE NOT NULL,
 PRIMARY KEY (trainid, gymid),
 FOREIGN KEY (trainid) REFERENCES Train(trainid),
 FOREIGN KEY (gymid) REFERENCES Gym(gymid));

CREATE TABLE TGSta
(tgsid VARCHAR(20) PRIMARY KEY,
 trainid VARCHAR(20) NOT NULL,
 tgstatus VARCHAR(20) NOT NULL,
 FOREIGN KEY (trainid) REFERENCES Train(trainid));

CREATE TABLE TE 
(mid VARCHAR(20) NOT NULL,
 trainid VARCHAR(20) NOT NULL,
 tetime DATE NOT NULL,
 testatus VARCHAR(20) NOT NULL,
 tescore INT,
 tefeedback VARCHAR(1000),
 PRIMARY KEY (mid, trainid),
 FOREIGN KEY (mid) REFERENCES Member(mid),
 FOREIGN KEY (trainid) REFERENCES Train(trainid));

CREATE TABLE Contest
(contestid VARCHAR(20) PRIMARY KEY,
contestname VARCHAR(20) NOT NULL,
contesttime DATE NOT NULL,
contestprize VARCHAR(50) NOT NULL,
contestnum INT NOT NULL);

CREATE TABLE CSta
(csid VARCHAR(20) PRIMARY KEY,
contestid VARCHAR(20) NOT NULL,
conteststatus VARCHAR(20) NOT NULL,
FOREIGN KEY (contestid) REFERENCES Contest(contestid));

CREATE TABLE CG
(contestid VARCHAR(20) NOT NULL,
gymid VARCHAR(20) NOT NULL,
cgtime DATE NOT NULL,
PRIMARY KEY (contestid, gymid),
FOREIGN KEY (contestid) REFERENCES Contest(contestid),
FOREIGN KEY (gymid) REFERENCES Gym(gymid));

CREATE TABLE CGSta
(cgsid VARCHAR(20) PRIMARY KEY,
 contestid VARCHAR(20) NOT NULL,
 cgstatus VARCHAR(20) NOT NULL,
 FOREIGN KEY (contestid) REFERENCES Contest(contestid));

CREATE TABLE CE
(mid VARCHAR(20) NOT NULL,
 contestid VARCHAR(20) NOT NULL,
 tetime DATE NOT NULL,
 testatus VARCHAR(20) NOT NULL,
 tescore INT,
 tefeedback VARCHAR(1000),
 PRIMARY KEY (mid, contestid),
 FOREIGN KEY (mid) REFERENCES Member(mid),
 FOREIGN KEY (contestid) REFERENCES Contest(contestid));

CREATE TABLE Sponsor
(sponsorid VARCHAR(20) PRIMARY KEY,
sponsorname VARCHAR(20) NOT NULL,
sponsortime DATE NOT NULL,
sponsorstatus VARCHAR(20) NOT NULL);

CREATE TABLE CS
(contestid VARCHAR(20) NOT NULL,
sponsorid VARCHAR(20) NOT NULL,
sponsorname VARCHAR(20) NOT NULL,
sponsortime DATE NOT NULL,
sponsorstatus VARCHAR(20) NOT NULL,
PRIMARY KEY (contestid, sponsorid),
FOREIGN KEY (contestid) REFERENCES Contest(contestid),
FOREIGN KEY (sponsorid) REFERENCES Sponsor(sponsorid));

CREATE TABLE Item
(itemid VARCHAR(20) PRIMARY KEY,
itemname VARCHAR(20) NOT NULL,
itemdescription VARCHAR(100) NOT NULL);

CREATE TABLE SI
(sponsorid VARCHAR(20) NOT NULL,
itemid VARCHAR(20) NOT NULL,
siquant INT NOT NULL,
sitime DATE NOT NULL,
PRIMARY KEY (sponsorid, itemid),
FOREIGN KEY (sponsorid) REFERENCES Sponsor(sponsorid),
FOREIGN KEY (itemid) REFERENCES Item(itemid));

CREATE TABLE MI
(mid VARCHAR(20) NOT NULL,
itemid VARCHAR(20) NOT NULL,
itemquant INT NOT NULL,
mitime DATE NOT NULL,
mistatus VARCHAR (20) NOT NULL,
PRIMARY KEY (mid, itemid),
FOREIGN KEY (mid) REFERENCES Member(mid),
FOREIGN KEY (itemid) REFERENCES Item(itemid));

CREATE TABLE Volunteer
(volunteerid VARCHAR(20) PRIMARY KEY,
volunteername VARCHAR(20) NOT NULL,
isadmin VARCHAR(10) NOT NULL);

CREATE TABLE VC
(volunteerid VARCHAR(20) NOT NULL,
contestid VARCHAR(20) NOT NULL,
vccontent VARCHAR(100) NOT NULL,
vcstatus VARCHAR(20) NOT NULL,
PRIMARY KEY (volunteerid, contestid),
FOREIGN KEY (volunteerid) REFERENCES Volunteer(volunteerid),
FOREIGN KEY (contestid) REFERENCES Contest(contestid));

CREATE TABLE Report
(reportid VARCHAR(20) PRIMARY KEY,
mid VARCHAR(20) NOT NULL,
reporttime DATE NOT NULL,
actfrequency VARCHAR(20) NOT NULL,
acttype VARCHAR(20),
actmile DECIMAL(5,2),
acthr INT,
bfp DECIMAL(4,2),
advice VARCHAR(1000) NOT NULL,
FOREIGN KEY (mid) REFERENCES Member(mid));




