
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

	orientation(lb1gt_3,134,1069). coords(lb1gt_3,162,33,1069).  visible(lb1gt_3,1069). happensAt(walking(lb1gt_3),1069). orientation(lb1gt_4,90,1069). coords(lb1gt_4,310,112,1069).  visible(lb1gt_4,1069). happensAt(inactive(lb1gt_4),1069).
orientation(lb1gt_3,134,1070). coords(lb1gt_3,163,33,1070).  visible(lb1gt_3,1070). happensAt(walking(lb1gt_3),1070). orientation(lb1gt_4,90,1070). coords(lb1gt_4,310,112,1070).  visible(lb1gt_4,1070). happensAt(inactive(lb1gt_4),1070).
orientation(lb1gt_3,134,1071). coords(lb1gt_3,164,32,1071).  visible(lb1gt_3,1071). happensAt(walking(lb1gt_3),1071). orientation(lb1gt_4,90,1071). coords(lb1gt_4,310,112,1071).  visible(lb1gt_4,1071). happensAt(inactive(lb1gt_4),1071).
orientation(lb1gt_3,134,1072). coords(lb1gt_3,165,32,1072).  visible(lb1gt_3,1072). happensAt(walking(lb1gt_3),1072). orientation(lb1gt_4,90,1072). coords(lb1gt_4,310,112,1072).  visible(lb1gt_4,1072). happensAt(inactive(lb1gt_4),1072).
orientation(lb1gt_3,134,1073). coords(lb1gt_3,166,32,1073).  visible(lb1gt_3,1073). happensAt(walking(lb1gt_3),1073). orientation(lb1gt_4,90,1073). coords(lb1gt_4,310,112,1073).  visible(lb1gt_4,1073). happensAt(inactive(lb1gt_4),1073).
orientation(lb1gt_3,134,1074). coords(lb1gt_3,167,32,1074).  visible(lb1gt_3,1074). happensAt(walking(lb1gt_3),1074). orientation(lb1gt_4,90,1074). coords(lb1gt_4,310,112,1074).  visible(lb1gt_4,1074). happensAt(inactive(lb1gt_4),1074).
orientation(lb1gt_3,134,1075). coords(lb1gt_3,168,31,1075).  visible(lb1gt_3,1075). happensAt(walking(lb1gt_3),1075). orientation(lb1gt_4,90,1075). coords(lb1gt_4,310,112,1075).  visible(lb1gt_4,1075). happensAt(inactive(lb1gt_4),1075).
orientation(lb1gt_3,134,1076). coords(lb1gt_3,168,30,1076).  visible(lb1gt_3,1076). happensAt(walking(lb1gt_3),1076). orientation(lb1gt_4,90,1076). coords(lb1gt_4,310,112,1076).  visible(lb1gt_4,1076). happensAt(inactive(lb1gt_4),1076).
orientation(lb1gt_3,134,1077). coords(lb1gt_3,168,30,1077).  visible(lb1gt_3,1077). happensAt(walking(lb1gt_3),1077). orientation(lb1gt_4,90,1077). coords(lb1gt_4,310,112,1077).  visible(lb1gt_4,1077). happensAt(inactive(lb1gt_4),1077).
orientation(lb1gt_3,134,1078). coords(lb1gt_3,169,30,1078).  visible(lb1gt_3,1078). happensAt(walking(lb1gt_3),1078). orientation(lb1gt_4,90,1078). coords(lb1gt_4,310,112,1078).  visible(lb1gt_4,1078). happensAt(inactive(lb1gt_4),1078).
orientation(lb1gt_3,134,1079). coords(lb1gt_3,170,30,1079).  visible(lb1gt_3,1079). happensAt(walking(lb1gt_3),1079). orientation(lb1gt_4,90,1079). coords(lb1gt_4,310,112,1079).  visible(lb1gt_4,1079). happensAt(inactive(lb1gt_4),1079).
orientation(lb1gt_3,134,1080). coords(lb1gt_3,170,29,1080).  visible(lb1gt_3,1080). happensAt(walking(lb1gt_3),1080). orientation(lb1gt_4,90,1080). coords(lb1gt_4,310,112,1080).  visible(lb1gt_4,1080). happensAt(inactive(lb1gt_4),1080).
orientation(lb1gt_3,134,1081). coords(lb1gt_3,170,29,1081).  visible(lb1gt_3,1081). happensAt(walking(lb1gt_3),1081). orientation(lb1gt_4,90,1081). coords(lb1gt_4,310,112,1081).  visible(lb1gt_4,1081). happensAt(inactive(lb1gt_4),1081).
orientation(lb1gt_3,134,1082). coords(lb1gt_3,171,28,1082).  visible(lb1gt_3,1082). happensAt(walking(lb1gt_3),1082). orientation(lb1gt_4,90,1082). coords(lb1gt_4,310,112,1082).  visible(lb1gt_4,1082). happensAt(inactive(lb1gt_4),1082).
orientation(lb1gt_3,134,1083). coords(lb1gt_3,171,28,1083).  visible(lb1gt_3,1083). happensAt(walking(lb1gt_3),1083). orientation(lb1gt_4,90,1083). coords(lb1gt_4,310,112,1083).  visible(lb1gt_4,1083). happensAt(inactive(lb1gt_4),1083).
orientation(lb1gt_3,134,1084). coords(lb1gt_3,172,28,1084).  visible(lb1gt_3,1084). happensAt(walking(lb1gt_3),1084). orientation(lb1gt_4,90,1084). coords(lb1gt_4,310,112,1084).  visible(lb1gt_4,1084). happensAt(inactive(lb1gt_4),1084).
orientation(lb1gt_3,134,1085). coords(lb1gt_3,172,28,1085).  visible(lb1gt_3,1085). happensAt(walking(lb1gt_3),1085). orientation(lb1gt_4,90,1085). coords(lb1gt_4,310,112,1085).  visible(lb1gt_4,1085). happensAt(inactive(lb1gt_4),1085).
orientation(lb1gt_3,134,1086). coords(lb1gt_3,173,28,1086).  visible(lb1gt_3,1086). happensAt(walking(lb1gt_3),1086). orientation(lb1gt_4,90,1086). coords(lb1gt_4,310,112,1086).  visible(lb1gt_4,1086). happensAt(inactive(lb1gt_4),1086).
orientation(lb1gt_3,134,1087). coords(lb1gt_3,172,27,1087).  visible(lb1gt_3,1087). happensAt(walking(lb1gt_3),1087). orientation(lb1gt_4,90,1087). coords(lb1gt_4,310,112,1087).  visible(lb1gt_4,1087). happensAt(inactive(lb1gt_4),1087).
orientation(lb1gt_3,134,1088). coords(lb1gt_3,173,26,1088).  visible(lb1gt_3,1088). happensAt(walking(lb1gt_3),1088). orientation(lb1gt_4,90,1088). coords(lb1gt_4,310,112,1088).  visible(lb1gt_4,1088). happensAt(inactive(lb1gt_4),1088).
orientation(lb1gt_3,134,1089). coords(lb1gt_3,174,26,1089).  visible(lb1gt_3,1089). happensAt(walking(lb1gt_3),1089). orientation(lb1gt_4,90,1089). coords(lb1gt_4,310,112,1089).  visible(lb1gt_4,1089). happensAt(inactive(lb1gt_4),1089).
orientation(lb1gt_3,134,1090). coords(lb1gt_3,172,22,1090).  visible(lb1gt_3,1090). happensAt(walking(lb1gt_3),1090). orientation(lb1gt_4,90,1090). coords(lb1gt_4,310,112,1090).  visible(lb1gt_4,1090). happensAt(inactive(lb1gt_4),1090).
orientation(lb1gt_3,134,1091). coords(lb1gt_3,172,22,1091).  visible(lb1gt_3,1091). happensAt(walking(lb1gt_3),1091). orientation(lb1gt_4,90,1091). coords(lb1gt_4,310,112,1091).  visible(lb1gt_4,1091). happensAt(inactive(lb1gt_4),1091).
orientation(lb1gt_3,134,1092). coords(lb1gt_3,173,22,1092).  visible(lb1gt_3,1092). happensAt(walking(lb1gt_3),1092). orientation(lb1gt_4,90,1092). coords(lb1gt_4,310,112,1092).  visible(lb1gt_4,1092). happensAt(inactive(lb1gt_4),1092).
orientation(lb1gt_3,134,1093). coords(lb1gt_3,174,23,1093).  happensAt(disappear(lb1gt_3),1093). happensAt(walking(lb1gt_3),1093). orientation(lb1gt_4,90,1093). coords(lb1gt_4,310,112,1093).  visible(lb1gt_4,1093). happensAt(inactive(lb1gt_4),1093).
#show holdsAt/2.