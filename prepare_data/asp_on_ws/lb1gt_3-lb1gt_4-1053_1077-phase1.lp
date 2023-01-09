
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

	orientation(lb1gt_3,134,1053). coords(lb1gt_3,155,46,1053).  visible(lb1gt_3,1053). happensAt(walking(lb1gt_3),1053). orientation(lb1gt_4,90,1053). coords(lb1gt_4,310,112,1053).  visible(lb1gt_4,1053). happensAt(inactive(lb1gt_4),1053).
orientation(lb1gt_3,134,1054). coords(lb1gt_3,155,45,1054).  visible(lb1gt_3,1054). happensAt(walking(lb1gt_3),1054). orientation(lb1gt_4,90,1054). coords(lb1gt_4,310,112,1054).  visible(lb1gt_4,1054). happensAt(inactive(lb1gt_4),1054).
orientation(lb1gt_3,134,1055). coords(lb1gt_3,156,43,1055).  visible(lb1gt_3,1055). happensAt(walking(lb1gt_3),1055). orientation(lb1gt_4,90,1055). coords(lb1gt_4,310,112,1055).  visible(lb1gt_4,1055). happensAt(inactive(lb1gt_4),1055).
orientation(lb1gt_3,134,1056). coords(lb1gt_3,156,41,1056).  visible(lb1gt_3,1056). happensAt(walking(lb1gt_3),1056). orientation(lb1gt_4,90,1056). coords(lb1gt_4,310,112,1056).  visible(lb1gt_4,1056). happensAt(inactive(lb1gt_4),1056).
orientation(lb1gt_3,134,1057). coords(lb1gt_3,156,40,1057).  visible(lb1gt_3,1057). happensAt(walking(lb1gt_3),1057). orientation(lb1gt_4,90,1057). coords(lb1gt_4,310,112,1057).  visible(lb1gt_4,1057). happensAt(inactive(lb1gt_4),1057).
orientation(lb1gt_3,134,1058). coords(lb1gt_3,157,39,1058).  visible(lb1gt_3,1058). happensAt(walking(lb1gt_3),1058). orientation(lb1gt_4,90,1058). coords(lb1gt_4,310,112,1058).  visible(lb1gt_4,1058). happensAt(inactive(lb1gt_4),1058).
orientation(lb1gt_3,134,1059). coords(lb1gt_3,158,38,1059).  visible(lb1gt_3,1059). happensAt(walking(lb1gt_3),1059). orientation(lb1gt_4,90,1059). coords(lb1gt_4,310,112,1059).  visible(lb1gt_4,1059). happensAt(inactive(lb1gt_4),1059).
orientation(lb1gt_3,134,1060). coords(lb1gt_3,159,38,1060).  visible(lb1gt_3,1060). happensAt(walking(lb1gt_3),1060). orientation(lb1gt_4,90,1060). coords(lb1gt_4,310,112,1060).  visible(lb1gt_4,1060). happensAt(inactive(lb1gt_4),1060).
orientation(lb1gt_3,134,1061). coords(lb1gt_3,160,36,1061).  visible(lb1gt_3,1061). happensAt(walking(lb1gt_3),1061). orientation(lb1gt_4,90,1061). coords(lb1gt_4,310,112,1061).  visible(lb1gt_4,1061). happensAt(inactive(lb1gt_4),1061).
orientation(lb1gt_3,134,1062). coords(lb1gt_3,160,36,1062).  visible(lb1gt_3,1062). happensAt(walking(lb1gt_3),1062). orientation(lb1gt_4,90,1062). coords(lb1gt_4,310,112,1062).  visible(lb1gt_4,1062). happensAt(inactive(lb1gt_4),1062).
orientation(lb1gt_3,134,1063). coords(lb1gt_3,161,35,1063).  visible(lb1gt_3,1063). happensAt(walking(lb1gt_3),1063). orientation(lb1gt_4,90,1063). coords(lb1gt_4,310,112,1063).  visible(lb1gt_4,1063). happensAt(inactive(lb1gt_4),1063).
orientation(lb1gt_3,134,1064). coords(lb1gt_3,160,35,1064).  visible(lb1gt_3,1064). happensAt(walking(lb1gt_3),1064). orientation(lb1gt_4,90,1064). coords(lb1gt_4,310,112,1064).  visible(lb1gt_4,1064). happensAt(inactive(lb1gt_4),1064).
orientation(lb1gt_3,134,1065). coords(lb1gt_3,161,35,1065).  visible(lb1gt_3,1065). happensAt(walking(lb1gt_3),1065). orientation(lb1gt_4,90,1065). coords(lb1gt_4,310,112,1065).  visible(lb1gt_4,1065). happensAt(inactive(lb1gt_4),1065).
orientation(lb1gt_3,134,1066). coords(lb1gt_3,161,34,1066).  visible(lb1gt_3,1066). happensAt(walking(lb1gt_3),1066). orientation(lb1gt_4,90,1066). coords(lb1gt_4,310,112,1066).  visible(lb1gt_4,1066). happensAt(inactive(lb1gt_4),1066).
orientation(lb1gt_3,134,1067). coords(lb1gt_3,161,34,1067).  visible(lb1gt_3,1067). happensAt(walking(lb1gt_3),1067). orientation(lb1gt_4,90,1067). coords(lb1gt_4,310,112,1067).  visible(lb1gt_4,1067). happensAt(inactive(lb1gt_4),1067).
orientation(lb1gt_3,134,1068). coords(lb1gt_3,161,34,1068).  visible(lb1gt_3,1068). happensAt(walking(lb1gt_3),1068). orientation(lb1gt_4,90,1068). coords(lb1gt_4,310,112,1068).  visible(lb1gt_4,1068). happensAt(inactive(lb1gt_4),1068).
orientation(lb1gt_3,134,1069). coords(lb1gt_3,162,33,1069).  visible(lb1gt_3,1069). happensAt(walking(lb1gt_3),1069). orientation(lb1gt_4,90,1069). coords(lb1gt_4,310,112,1069).  visible(lb1gt_4,1069). happensAt(inactive(lb1gt_4),1069).
orientation(lb1gt_3,134,1070). coords(lb1gt_3,163,33,1070).  visible(lb1gt_3,1070). happensAt(walking(lb1gt_3),1070). orientation(lb1gt_4,90,1070). coords(lb1gt_4,310,112,1070).  visible(lb1gt_4,1070). happensAt(inactive(lb1gt_4),1070).
orientation(lb1gt_3,134,1071). coords(lb1gt_3,164,32,1071).  visible(lb1gt_3,1071). happensAt(walking(lb1gt_3),1071). orientation(lb1gt_4,90,1071). coords(lb1gt_4,310,112,1071).  visible(lb1gt_4,1071). happensAt(inactive(lb1gt_4),1071).
orientation(lb1gt_3,134,1072). coords(lb1gt_3,165,32,1072).  visible(lb1gt_3,1072). happensAt(walking(lb1gt_3),1072). orientation(lb1gt_4,90,1072). coords(lb1gt_4,310,112,1072).  visible(lb1gt_4,1072). happensAt(inactive(lb1gt_4),1072).
orientation(lb1gt_3,134,1073). coords(lb1gt_3,166,32,1073).  visible(lb1gt_3,1073). happensAt(walking(lb1gt_3),1073). orientation(lb1gt_4,90,1073). coords(lb1gt_4,310,112,1073).  visible(lb1gt_4,1073). happensAt(inactive(lb1gt_4),1073).
orientation(lb1gt_3,134,1074). coords(lb1gt_3,167,32,1074).  visible(lb1gt_3,1074). happensAt(walking(lb1gt_3),1074). orientation(lb1gt_4,90,1074). coords(lb1gt_4,310,112,1074).  visible(lb1gt_4,1074). happensAt(inactive(lb1gt_4),1074).
orientation(lb1gt_3,134,1075). coords(lb1gt_3,168,31,1075).  visible(lb1gt_3,1075). happensAt(walking(lb1gt_3),1075). orientation(lb1gt_4,90,1075). coords(lb1gt_4,310,112,1075).  visible(lb1gt_4,1075). happensAt(inactive(lb1gt_4),1075).
orientation(lb1gt_3,134,1076). coords(lb1gt_3,168,30,1076).  visible(lb1gt_3,1076). happensAt(walking(lb1gt_3),1076). orientation(lb1gt_4,90,1076). coords(lb1gt_4,310,112,1076).  visible(lb1gt_4,1076). happensAt(inactive(lb1gt_4),1076).
orientation(lb1gt_3,134,1077). coords(lb1gt_3,168,30,1077).  visible(lb1gt_3,1077). happensAt(walking(lb1gt_3),1077). orientation(lb1gt_4,90,1077). coords(lb1gt_4,310,112,1077).  visible(lb1gt_4,1077). happensAt(inactive(lb1gt_4),1077).
#show holdsAt/2.