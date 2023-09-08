
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

	orientation(lb1gt_3,134,1041). coords(lb1gt_3,155,60,1041).  visible(lb1gt_3,1041). happensAt(walking(lb1gt_3),1041). orientation(lb1gt_4,90,1041). coords(lb1gt_4,310,112,1041).  visible(lb1gt_4,1041). happensAt(inactive(lb1gt_4),1041).
orientation(lb1gt_3,134,1042). coords(lb1gt_3,156,59,1042).  visible(lb1gt_3,1042). happensAt(walking(lb1gt_3),1042). orientation(lb1gt_4,90,1042). coords(lb1gt_4,310,112,1042).  visible(lb1gt_4,1042). happensAt(inactive(lb1gt_4),1042).
orientation(lb1gt_3,134,1043). coords(lb1gt_3,156,58,1043).  visible(lb1gt_3,1043). happensAt(walking(lb1gt_3),1043). orientation(lb1gt_4,90,1043). coords(lb1gt_4,310,112,1043).  visible(lb1gt_4,1043). happensAt(inactive(lb1gt_4),1043).
orientation(lb1gt_3,134,1044). coords(lb1gt_3,155,57,1044).  visible(lb1gt_3,1044). happensAt(walking(lb1gt_3),1044). orientation(lb1gt_4,90,1044). coords(lb1gt_4,310,112,1044).  visible(lb1gt_4,1044). happensAt(inactive(lb1gt_4),1044).
orientation(lb1gt_3,134,1045). coords(lb1gt_3,155,56,1045).  visible(lb1gt_3,1045). happensAt(walking(lb1gt_3),1045). orientation(lb1gt_4,90,1045). coords(lb1gt_4,310,112,1045).  visible(lb1gt_4,1045). happensAt(inactive(lb1gt_4),1045).
orientation(lb1gt_3,134,1046). coords(lb1gt_3,155,54,1046).  visible(lb1gt_3,1046). happensAt(walking(lb1gt_3),1046). orientation(lb1gt_4,90,1046). coords(lb1gt_4,310,112,1046).  visible(lb1gt_4,1046). happensAt(inactive(lb1gt_4),1046).
orientation(lb1gt_3,134,1047). coords(lb1gt_3,155,52,1047).  visible(lb1gt_3,1047). happensAt(walking(lb1gt_3),1047). orientation(lb1gt_4,90,1047). coords(lb1gt_4,310,112,1047).  visible(lb1gt_4,1047). happensAt(inactive(lb1gt_4),1047).
orientation(lb1gt_3,134,1048). coords(lb1gt_3,155,50,1048).  visible(lb1gt_3,1048). happensAt(walking(lb1gt_3),1048). orientation(lb1gt_4,90,1048). coords(lb1gt_4,310,112,1048).  visible(lb1gt_4,1048). happensAt(inactive(lb1gt_4),1048).
orientation(lb1gt_3,134,1049). coords(lb1gt_3,155,49,1049).  visible(lb1gt_3,1049). happensAt(walking(lb1gt_3),1049). orientation(lb1gt_4,90,1049). coords(lb1gt_4,310,112,1049).  visible(lb1gt_4,1049). happensAt(inactive(lb1gt_4),1049).
orientation(lb1gt_3,134,1050). coords(lb1gt_3,155,48,1050).  visible(lb1gt_3,1050). happensAt(walking(lb1gt_3),1050). orientation(lb1gt_4,90,1050). coords(lb1gt_4,310,112,1050).  visible(lb1gt_4,1050). happensAt(inactive(lb1gt_4),1050).
orientation(lb1gt_3,134,1051). coords(lb1gt_3,155,47,1051).  visible(lb1gt_3,1051). happensAt(walking(lb1gt_3),1051). orientation(lb1gt_4,90,1051). coords(lb1gt_4,310,112,1051).  visible(lb1gt_4,1051). happensAt(inactive(lb1gt_4),1051).
orientation(lb1gt_3,134,1052). coords(lb1gt_3,155,47,1052).  visible(lb1gt_3,1052). happensAt(walking(lb1gt_3),1052). orientation(lb1gt_4,90,1052). coords(lb1gt_4,310,112,1052).  visible(lb1gt_4,1052). happensAt(inactive(lb1gt_4),1052).
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
#show holdsAt/2.