
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

	orientation(lb1gt_4,90,1216). coords(lb1gt_4,310,112,1216).  visible(lb1gt_4,1216). happensAt(inactive(lb1gt_4),1216). orientation(lb1gt_5,117,1216). coords(lb1gt_5,174,26,1216).  happensAt(appear(lb1gt_5),1216). happensAt(walking(lb1gt_5),1216).
orientation(lb1gt_4,90,1217). coords(lb1gt_4,310,112,1217).  visible(lb1gt_4,1217). happensAt(inactive(lb1gt_4),1217). orientation(lb1gt_5,117,1217). coords(lb1gt_5,174,26,1217).  visible(lb1gt_5,1217). happensAt(walking(lb1gt_5),1217).
orientation(lb1gt_4,90,1218). coords(lb1gt_4,310,112,1218).  visible(lb1gt_4,1218). happensAt(inactive(lb1gt_4),1218). orientation(lb1gt_5,117,1218). coords(lb1gt_5,174,26,1218).  visible(lb1gt_5,1218). happensAt(walking(lb1gt_5),1218).
orientation(lb1gt_4,90,1219). coords(lb1gt_4,310,112,1219).  visible(lb1gt_4,1219). happensAt(inactive(lb1gt_4),1219). orientation(lb1gt_5,117,1219). coords(lb1gt_5,174,27,1219).  visible(lb1gt_5,1219). happensAt(walking(lb1gt_5),1219).
orientation(lb1gt_4,90,1220). coords(lb1gt_4,310,112,1220).  visible(lb1gt_4,1220). happensAt(inactive(lb1gt_4),1220). orientation(lb1gt_5,117,1220). coords(lb1gt_5,175,28,1220).  visible(lb1gt_5,1220). happensAt(walking(lb1gt_5),1220).
orientation(lb1gt_4,90,1221). coords(lb1gt_4,310,112,1221).  visible(lb1gt_4,1221). happensAt(inactive(lb1gt_4),1221). orientation(lb1gt_5,117,1221). coords(lb1gt_5,174,29,1221).  visible(lb1gt_5,1221). happensAt(walking(lb1gt_5),1221).
orientation(lb1gt_4,90,1222). coords(lb1gt_4,310,112,1222).  visible(lb1gt_4,1222). happensAt(inactive(lb1gt_4),1222). orientation(lb1gt_5,117,1222). coords(lb1gt_5,176,31,1222).  visible(lb1gt_5,1222). happensAt(walking(lb1gt_5),1222).
orientation(lb1gt_4,90,1223). coords(lb1gt_4,310,112,1223).  visible(lb1gt_4,1223). happensAt(inactive(lb1gt_4),1223). orientation(lb1gt_5,117,1223). coords(lb1gt_5,176,33,1223).  visible(lb1gt_5,1223). happensAt(walking(lb1gt_5),1223).
orientation(lb1gt_4,90,1224). coords(lb1gt_4,310,112,1224).  visible(lb1gt_4,1224). happensAt(inactive(lb1gt_4),1224). orientation(lb1gt_5,117,1224). coords(lb1gt_5,176,34,1224).  visible(lb1gt_5,1224). happensAt(walking(lb1gt_5),1224).
orientation(lb1gt_4,90,1225). coords(lb1gt_4,310,112,1225).  visible(lb1gt_4,1225). happensAt(inactive(lb1gt_4),1225). orientation(lb1gt_5,117,1225). coords(lb1gt_5,176,35,1225).  visible(lb1gt_5,1225). happensAt(walking(lb1gt_5),1225).
orientation(lb1gt_4,90,1226). coords(lb1gt_4,310,112,1226).  visible(lb1gt_4,1226). happensAt(inactive(lb1gt_4),1226). orientation(lb1gt_5,117,1226). coords(lb1gt_5,175,36,1226).  visible(lb1gt_5,1226). happensAt(walking(lb1gt_5),1226).
orientation(lb1gt_4,90,1227). coords(lb1gt_4,310,112,1227).  visible(lb1gt_4,1227). happensAt(inactive(lb1gt_4),1227). orientation(lb1gt_5,119,1227). coords(lb1gt_5,175,36,1227).  visible(lb1gt_5,1227). happensAt(walking(lb1gt_5),1227).
orientation(lb1gt_4,90,1228). coords(lb1gt_4,310,112,1228).  visible(lb1gt_4,1228). happensAt(inactive(lb1gt_4),1228). orientation(lb1gt_5,122,1228). coords(lb1gt_5,174,36,1228).  visible(lb1gt_5,1228). happensAt(walking(lb1gt_5),1228).
orientation(lb1gt_4,90,1229). coords(lb1gt_4,310,112,1229).  visible(lb1gt_4,1229). happensAt(inactive(lb1gt_4),1229). orientation(lb1gt_5,122,1229). coords(lb1gt_5,174,36,1229).  visible(lb1gt_5,1229). happensAt(walking(lb1gt_5),1229).
orientation(lb1gt_4,90,1230). coords(lb1gt_4,310,112,1230).  visible(lb1gt_4,1230). happensAt(inactive(lb1gt_4),1230). orientation(lb1gt_5,125,1230). coords(lb1gt_5,173,37,1230).  visible(lb1gt_5,1230). happensAt(walking(lb1gt_5),1230).
orientation(lb1gt_4,90,1231). coords(lb1gt_4,310,112,1231).  visible(lb1gt_4,1231). happensAt(inactive(lb1gt_4),1231). orientation(lb1gt_5,124,1231). coords(lb1gt_5,173,37,1231).  visible(lb1gt_5,1231). happensAt(walking(lb1gt_5),1231).
orientation(lb1gt_4,90,1232). coords(lb1gt_4,310,112,1232).  visible(lb1gt_4,1232). happensAt(inactive(lb1gt_4),1232). orientation(lb1gt_5,123,1232). coords(lb1gt_5,173,37,1232).  visible(lb1gt_5,1232). happensAt(walking(lb1gt_5),1232).
orientation(lb1gt_4,90,1233). coords(lb1gt_4,310,112,1233).  visible(lb1gt_4,1233). happensAt(inactive(lb1gt_4),1233). orientation(lb1gt_5,124,1233). coords(lb1gt_5,173,37,1233).  visible(lb1gt_5,1233). happensAt(walking(lb1gt_5),1233).
orientation(lb1gt_4,90,1234). coords(lb1gt_4,310,112,1234).  visible(lb1gt_4,1234). happensAt(inactive(lb1gt_4),1234). orientation(lb1gt_5,123,1234). coords(lb1gt_5,172,37,1234).  visible(lb1gt_5,1234). happensAt(walking(lb1gt_5),1234).
orientation(lb1gt_4,90,1235). coords(lb1gt_4,310,112,1235).  visible(lb1gt_4,1235). happensAt(inactive(lb1gt_4),1235). orientation(lb1gt_5,124,1235). coords(lb1gt_5,172,37,1235).  visible(lb1gt_5,1235). happensAt(walking(lb1gt_5),1235).
orientation(lb1gt_4,90,1236). coords(lb1gt_4,310,112,1236).  visible(lb1gt_4,1236). happensAt(inactive(lb1gt_4),1236). orientation(lb1gt_5,127,1236). coords(lb1gt_5,172,38,1236).  visible(lb1gt_5,1236). happensAt(walking(lb1gt_5),1236).
orientation(lb1gt_4,90,1237). coords(lb1gt_4,310,112,1237).  visible(lb1gt_4,1237). happensAt(inactive(lb1gt_4),1237). orientation(lb1gt_5,127,1237). coords(lb1gt_5,173,38,1237).  visible(lb1gt_5,1237). happensAt(walking(lb1gt_5),1237).
orientation(lb1gt_4,90,1238). coords(lb1gt_4,310,112,1238).  visible(lb1gt_4,1238). happensAt(inactive(lb1gt_4),1238). orientation(lb1gt_5,127,1238). coords(lb1gt_5,174,40,1238).  visible(lb1gt_5,1238). happensAt(walking(lb1gt_5),1238).
orientation(lb1gt_4,90,1239). coords(lb1gt_4,310,112,1239).  visible(lb1gt_4,1239). happensAt(inactive(lb1gt_4),1239). orientation(lb1gt_5,127,1239). coords(lb1gt_5,174,41,1239).  visible(lb1gt_5,1239). happensAt(walking(lb1gt_5),1239).
orientation(lb1gt_4,90,1240). coords(lb1gt_4,310,112,1240).  visible(lb1gt_4,1240). happensAt(inactive(lb1gt_4),1240). orientation(lb1gt_5,127,1240). coords(lb1gt_5,174,42,1240).  visible(lb1gt_5,1240). happensAt(walking(lb1gt_5),1240).
#show holdsAt/2.