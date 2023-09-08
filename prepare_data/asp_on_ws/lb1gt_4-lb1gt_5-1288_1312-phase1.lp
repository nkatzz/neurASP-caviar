
holdsAt(F,Te) :- initiatedAt(F,Ts), fluent(F), next(Ts,Te).
holdsAt(F,Te) :- holdsAt(F,Ts), not terminatedAt(F,Ts), fluent(F), next(Ts,Te).


fluent(moving(X,Y)) :- person(X), person(Y), X != Y.
fluent(meeting(X,Y)) :- person(X), person(Y), X != Y.


#include "../next.lp".

% Distance/orientation calculations:
#include "../distances.lp".

% Definitions of "type" predicates (extracted from the data):
time(T) :- happensAt(disappear(_),T).
time(T) :- happensAt(appear(_),T).
time(T) :- happensAt(active(_),T).
time(T) :- happensAt(inactive(_),T).
time(T) :- happensAt(walking(_),T).
time(T) :- happensAt(running(_),T).
time(T) :- coords(_,_,_,T).
time(T) :- orientation(_,_,T).


person(X) :- happensAt(disappear(X),_).
person(X) :- happensAt(appear(X),_).
person(X) :- happensAt(active(X),_).
person(X) :- happensAt(inactive(X),_).
person(X) :- happensAt(walking(X),_).
person(X) :- happensAt(running(X),_).
person(X) :- coords(X,_,_,_).
person(X) :- orientation(X,_,_).


% Definition for the "moving" complex event: 
initiatedAt(moving(X0,X1),X2) :- happensAt(walking(X0),X2),happensAt(walking(X1),X2),orientationMove(X0,X1,X2),close(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(active(X1),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(inactive(X1),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X1),X2),happensAt(inactive(X0),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(running(X0),X2), person(X1).
terminatedAt(moving(X0,X1),X2) :- happensAt(running(X1),X2), person(X0).
terminatedAt(moving(X0,X1),X2) :- happensAt(disappear(X0),X2), person(X1).
terminatedAt(moving(X0,X1),X2) :- happensAt(disappear(X1),X2), person(X0).

% Definition for the "meeting" complex event:
initiatedAt(meeting(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(active(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(inactive(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(inactive(X0),X2),happensAt(active(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(inactive(X0),X2),happensAt(inactive(X1),X2),close(X0,X1,25,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(running(X0),X2),person(X1).
terminatedAt(meeting(X0,X1),X2) :- happensAt(running(X1),X2),person(X0).
terminatedAt(meeting(X0,X1),X2) :- happensAt(disappear(X0),X2), person(X1).
terminatedAt(meeting(X0,X1),X2) :- happensAt(disappear(X1),X2), person(X0).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,25,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,25,X2).

	orientation(lb1gt_4,90,1288). coords(lb1gt_4,310,112,1288).  visible(lb1gt_4,1288). happensAt(inactive(lb1gt_4),1288). orientation(lb1gt_5,133,1288). coords(lb1gt_5,207,97,1288).  visible(lb1gt_5,1288). happensAt(walking(lb1gt_5),1288).
orientation(lb1gt_4,90,1289). coords(lb1gt_4,310,112,1289).  visible(lb1gt_4,1289). happensAt(inactive(lb1gt_4),1289). orientation(lb1gt_5,133,1289). coords(lb1gt_5,209,98,1289).  visible(lb1gt_5,1289). happensAt(walking(lb1gt_5),1289).
orientation(lb1gt_4,90,1290). coords(lb1gt_4,310,112,1290).  visible(lb1gt_4,1290). happensAt(inactive(lb1gt_4),1290). orientation(lb1gt_5,133,1290). coords(lb1gt_5,211,98,1290).  visible(lb1gt_5,1290). happensAt(walking(lb1gt_5),1290).
orientation(lb1gt_4,90,1291). coords(lb1gt_4,310,112,1291).  visible(lb1gt_4,1291). happensAt(inactive(lb1gt_4),1291). orientation(lb1gt_5,133,1291). coords(lb1gt_5,213,98,1291).  visible(lb1gt_5,1291). happensAt(walking(lb1gt_5),1291).
orientation(lb1gt_4,90,1292). coords(lb1gt_4,310,112,1292).  visible(lb1gt_4,1292). happensAt(inactive(lb1gt_4),1292). orientation(lb1gt_5,133,1292). coords(lb1gt_5,213,98,1292).  visible(lb1gt_5,1292). happensAt(walking(lb1gt_5),1292).
orientation(lb1gt_4,90,1293). coords(lb1gt_4,310,112,1293).  visible(lb1gt_4,1293). happensAt(inactive(lb1gt_4),1293). orientation(lb1gt_5,133,1293). coords(lb1gt_5,214,98,1293).  visible(lb1gt_5,1293). happensAt(walking(lb1gt_5),1293).
orientation(lb1gt_4,90,1294). coords(lb1gt_4,310,112,1294).  visible(lb1gt_4,1294). happensAt(inactive(lb1gt_4),1294). orientation(lb1gt_5,133,1294). coords(lb1gt_5,216,98,1294).  visible(lb1gt_5,1294). happensAt(walking(lb1gt_5),1294).
orientation(lb1gt_4,90,1295). coords(lb1gt_4,310,112,1295).  visible(lb1gt_4,1295). happensAt(inactive(lb1gt_4),1295). orientation(lb1gt_5,133,1295). coords(lb1gt_5,217,99,1295).  visible(lb1gt_5,1295). happensAt(walking(lb1gt_5),1295).
orientation(lb1gt_4,90,1296). coords(lb1gt_4,310,112,1296).  visible(lb1gt_4,1296). happensAt(inactive(lb1gt_4),1296). orientation(lb1gt_5,133,1296). coords(lb1gt_5,218,100,1296).  visible(lb1gt_5,1296). happensAt(walking(lb1gt_5),1296).
orientation(lb1gt_4,90,1297). coords(lb1gt_4,310,112,1297).  visible(lb1gt_4,1297). happensAt(inactive(lb1gt_4),1297). orientation(lb1gt_5,133,1297). coords(lb1gt_5,219,100,1297).  visible(lb1gt_5,1297). happensAt(walking(lb1gt_5),1297).
orientation(lb1gt_4,90,1298). coords(lb1gt_4,310,112,1298).  visible(lb1gt_4,1298). happensAt(inactive(lb1gt_4),1298). orientation(lb1gt_5,133,1298). coords(lb1gt_5,221,100,1298).  visible(lb1gt_5,1298). happensAt(walking(lb1gt_5),1298).
orientation(lb1gt_4,90,1299). coords(lb1gt_4,310,112,1299).  visible(lb1gt_4,1299). happensAt(inactive(lb1gt_4),1299). orientation(lb1gt_5,133,1299). coords(lb1gt_5,226,99,1299).  visible(lb1gt_5,1299). happensAt(walking(lb1gt_5),1299).
orientation(lb1gt_4,90,1300). coords(lb1gt_4,310,112,1300).  visible(lb1gt_4,1300). happensAt(inactive(lb1gt_4),1300). orientation(lb1gt_5,133,1300). coords(lb1gt_5,231,99,1300).  visible(lb1gt_5,1300). happensAt(walking(lb1gt_5),1300).
orientation(lb1gt_4,90,1301). coords(lb1gt_4,310,112,1301).  visible(lb1gt_4,1301). happensAt(inactive(lb1gt_4),1301). orientation(lb1gt_5,133,1301). coords(lb1gt_5,235,98,1301).  visible(lb1gt_5,1301). happensAt(walking(lb1gt_5),1301).
orientation(lb1gt_4,90,1302). coords(lb1gt_4,310,112,1302).  visible(lb1gt_4,1302). happensAt(inactive(lb1gt_4),1302). orientation(lb1gt_5,133,1302). coords(lb1gt_5,239,98,1302).  visible(lb1gt_5,1302). happensAt(walking(lb1gt_5),1302).
orientation(lb1gt_4,90,1303). coords(lb1gt_4,310,112,1303).  visible(lb1gt_4,1303). happensAt(inactive(lb1gt_4),1303). orientation(lb1gt_5,133,1303). coords(lb1gt_5,241,97,1303).  visible(lb1gt_5,1303). happensAt(walking(lb1gt_5),1303).
orientation(lb1gt_4,90,1304). coords(lb1gt_4,310,112,1304).  visible(lb1gt_4,1304). happensAt(inactive(lb1gt_4),1304). orientation(lb1gt_5,133,1304). coords(lb1gt_5,244,97,1304).  visible(lb1gt_5,1304). happensAt(walking(lb1gt_5),1304).
orientation(lb1gt_4,90,1305). coords(lb1gt_4,310,112,1305).  visible(lb1gt_4,1305). happensAt(inactive(lb1gt_4),1305). orientation(lb1gt_5,133,1305). coords(lb1gt_5,246,96,1305).  visible(lb1gt_5,1305). happensAt(walking(lb1gt_5),1305).
orientation(lb1gt_4,90,1306). coords(lb1gt_4,310,112,1306).  visible(lb1gt_4,1306). happensAt(inactive(lb1gt_4),1306). orientation(lb1gt_5,133,1306). coords(lb1gt_5,248,96,1306).  visible(lb1gt_5,1306). happensAt(walking(lb1gt_5),1306).
orientation(lb1gt_4,90,1307). coords(lb1gt_4,310,112,1307).  visible(lb1gt_4,1307). happensAt(inactive(lb1gt_4),1307). orientation(lb1gt_5,133,1307). coords(lb1gt_5,250,95,1307).  visible(lb1gt_5,1307). happensAt(walking(lb1gt_5),1307).
orientation(lb1gt_4,90,1308). coords(lb1gt_4,310,112,1308).  visible(lb1gt_4,1308). happensAt(inactive(lb1gt_4),1308). orientation(lb1gt_5,133,1308). coords(lb1gt_5,251,95,1308).  visible(lb1gt_5,1308). happensAt(walking(lb1gt_5),1308).
orientation(lb1gt_4,90,1309). coords(lb1gt_4,310,112,1309).  visible(lb1gt_4,1309). happensAt(inactive(lb1gt_4),1309). orientation(lb1gt_5,133,1309). coords(lb1gt_5,252,94,1309).  visible(lb1gt_5,1309). happensAt(walking(lb1gt_5),1309).
orientation(lb1gt_4,90,1310). coords(lb1gt_4,310,112,1310).  visible(lb1gt_4,1310). happensAt(inactive(lb1gt_4),1310). orientation(lb1gt_5,133,1310). coords(lb1gt_5,253,94,1310).  visible(lb1gt_5,1310). happensAt(walking(lb1gt_5),1310).
orientation(lb1gt_4,90,1311). coords(lb1gt_4,310,112,1311).  visible(lb1gt_4,1311). happensAt(inactive(lb1gt_4),1311). orientation(lb1gt_5,133,1311). coords(lb1gt_5,254,94,1311).  visible(lb1gt_5,1311). happensAt(walking(lb1gt_5),1311).
orientation(lb1gt_4,90,1312). coords(lb1gt_4,310,112,1312).  visible(lb1gt_4,1312). happensAt(inactive(lb1gt_4),1312). orientation(lb1gt_5,133,1312). coords(lb1gt_5,256,93,1312).  visible(lb1gt_5,1312). happensAt(walking(lb1gt_5),1312).
#show holdsAt/2.