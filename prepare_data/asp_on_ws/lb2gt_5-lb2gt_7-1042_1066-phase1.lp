
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

	orientation(lb2gt_5,152,1042). coords(lb2gt_5,177,169,1042).  visible(lb2gt_5,1042). happensAt(walking(lb2gt_5),1042). orientation(lb2gt_7,0,1042). coords(lb2gt_7,88,266,1042).  visible(lb2gt_7,1042). happensAt(walking(lb2gt_7),1042).
orientation(lb2gt_5,152,1043). coords(lb2gt_5,181,170,1043).  visible(lb2gt_5,1043). happensAt(walking(lb2gt_5),1043). orientation(lb2gt_7,0,1043). coords(lb2gt_7,88,266,1043).  visible(lb2gt_7,1043). happensAt(walking(lb2gt_7),1043).
orientation(lb2gt_5,152,1044). coords(lb2gt_5,183,174,1044).  visible(lb2gt_5,1044). happensAt(walking(lb2gt_5),1044). orientation(lb2gt_7,0,1044). coords(lb2gt_7,88,266,1044).  visible(lb2gt_7,1044). happensAt(walking(lb2gt_7),1044).
orientation(lb2gt_5,152,1045). coords(lb2gt_5,184,176,1045).  visible(lb2gt_5,1045). happensAt(walking(lb2gt_5),1045). orientation(lb2gt_7,0,1045). coords(lb2gt_7,88,267,1045).  visible(lb2gt_7,1045). happensAt(walking(lb2gt_7),1045).
orientation(lb2gt_5,152,1046). coords(lb2gt_5,186,178,1046).  visible(lb2gt_5,1046). happensAt(walking(lb2gt_5),1046). orientation(lb2gt_7,0,1046). coords(lb2gt_7,88,267,1046).  visible(lb2gt_7,1046). happensAt(walking(lb2gt_7),1046).
orientation(lb2gt_5,152,1047). coords(lb2gt_5,187,180,1047).  visible(lb2gt_5,1047). happensAt(walking(lb2gt_5),1047). orientation(lb2gt_7,0,1047). coords(lb2gt_7,87,267,1047).  visible(lb2gt_7,1047). happensAt(walking(lb2gt_7),1047).
orientation(lb2gt_5,151,1048). coords(lb2gt_5,189,182,1048).  visible(lb2gt_5,1048). happensAt(walking(lb2gt_5),1048). orientation(lb2gt_7,0,1048). coords(lb2gt_7,87,266,1048).  visible(lb2gt_7,1048). happensAt(walking(lb2gt_7),1048).
orientation(lb2gt_5,152,1049). coords(lb2gt_5,189,183,1049).  visible(lb2gt_5,1049). happensAt(walking(lb2gt_5),1049). orientation(lb2gt_7,0,1049). coords(lb2gt_7,87,266,1049).  visible(lb2gt_7,1049). happensAt(walking(lb2gt_7),1049).
orientation(lb2gt_5,152,1050). coords(lb2gt_5,190,184,1050).  visible(lb2gt_5,1050). happensAt(walking(lb2gt_5),1050). orientation(lb2gt_7,0,1050). coords(lb2gt_7,87,266,1050).  visible(lb2gt_7,1050). happensAt(walking(lb2gt_7),1050).
orientation(lb2gt_5,152,1051). coords(lb2gt_5,190,185,1051).  visible(lb2gt_5,1051). happensAt(walking(lb2gt_5),1051). orientation(lb2gt_7,160,1051). coords(lb2gt_7,87,266,1051).  visible(lb2gt_7,1051). happensAt(walking(lb2gt_7),1051).
orientation(lb2gt_5,152,1052). coords(lb2gt_5,191,185,1052).  visible(lb2gt_5,1052). happensAt(walking(lb2gt_5),1052). orientation(lb2gt_7,160,1052). coords(lb2gt_7,87,266,1052).  visible(lb2gt_7,1052). happensAt(walking(lb2gt_7),1052).
orientation(lb2gt_5,152,1053). coords(lb2gt_5,191,187,1053).  visible(lb2gt_5,1053). happensAt(walking(lb2gt_5),1053). orientation(lb2gt_7,160,1053). coords(lb2gt_7,87,266,1053).  visible(lb2gt_7,1053). happensAt(walking(lb2gt_7),1053).
orientation(lb2gt_5,165,1054). coords(lb2gt_5,191,188,1054).  visible(lb2gt_5,1054). happensAt(walking(lb2gt_5),1054). orientation(lb2gt_7,160,1054). coords(lb2gt_7,86,266,1054).  visible(lb2gt_7,1054). happensAt(walking(lb2gt_7),1054).
orientation(lb2gt_5,170,1055). coords(lb2gt_5,192,190,1055).  visible(lb2gt_5,1055). happensAt(walking(lb2gt_5),1055). orientation(lb2gt_7,160,1055). coords(lb2gt_7,86,266,1055).  visible(lb2gt_7,1055). happensAt(walking(lb2gt_7),1055).
orientation(lb2gt_5,164,1056). coords(lb2gt_5,193,192,1056).  visible(lb2gt_5,1056). happensAt(walking(lb2gt_5),1056). orientation(lb2gt_7,160,1056). coords(lb2gt_7,86,266,1056).  visible(lb2gt_7,1056). happensAt(walking(lb2gt_7),1056).
orientation(lb2gt_5,166,1057). coords(lb2gt_5,196,196,1057).  visible(lb2gt_5,1057). happensAt(walking(lb2gt_5),1057). orientation(lb2gt_7,160,1057). coords(lb2gt_7,85,264,1057).  visible(lb2gt_7,1057). happensAt(walking(lb2gt_7),1057).
orientation(lb2gt_5,164,1058). coords(lb2gt_5,198,198,1058).  visible(lb2gt_5,1058). happensAt(walking(lb2gt_5),1058). orientation(lb2gt_7,160,1058). coords(lb2gt_7,84,264,1058).  visible(lb2gt_7,1058). happensAt(walking(lb2gt_7),1058).
orientation(lb2gt_5,171,1059). coords(lb2gt_5,199,201,1059).  visible(lb2gt_5,1059). happensAt(walking(lb2gt_5),1059). orientation(lb2gt_7,160,1059). coords(lb2gt_7,83,263,1059).  visible(lb2gt_7,1059). happensAt(walking(lb2gt_7),1059).
orientation(lb2gt_5,174,1060). coords(lb2gt_5,201,203,1060).  visible(lb2gt_5,1060). happensAt(walking(lb2gt_5),1060). orientation(lb2gt_7,160,1060). coords(lb2gt_7,83,263,1060).  visible(lb2gt_7,1060). happensAt(walking(lb2gt_7),1060).
orientation(lb2gt_5,0,1061). coords(lb2gt_5,203,204,1061).  visible(lb2gt_5,1061). happensAt(walking(lb2gt_5),1061). orientation(lb2gt_7,160,1061). coords(lb2gt_7,83,262,1061).  visible(lb2gt_7,1061). happensAt(walking(lb2gt_7),1061).
orientation(lb2gt_5,3,1062). coords(lb2gt_5,204,205,1062).  visible(lb2gt_5,1062). happensAt(walking(lb2gt_5),1062). orientation(lb2gt_7,160,1062). coords(lb2gt_7,81,262,1062).  visible(lb2gt_7,1062). happensAt(walking(lb2gt_7),1062).
orientation(lb2gt_5,0,1063). coords(lb2gt_5,205,206,1063).  visible(lb2gt_5,1063). happensAt(walking(lb2gt_5),1063). orientation(lb2gt_7,160,1063). coords(lb2gt_7,80,262,1063).  visible(lb2gt_7,1063). happensAt(walking(lb2gt_7),1063).
orientation(lb2gt_5,169,1064). coords(lb2gt_5,205,206,1064).  visible(lb2gt_5,1064). happensAt(walking(lb2gt_5),1064). orientation(lb2gt_7,160,1064). coords(lb2gt_7,80,262,1064).  visible(lb2gt_7,1064). happensAt(walking(lb2gt_7),1064).
orientation(lb2gt_5,169,1065). coords(lb2gt_5,206,208,1065).  visible(lb2gt_5,1065). happensAt(walking(lb2gt_5),1065). orientation(lb2gt_7,160,1065). coords(lb2gt_7,80,261,1065).  visible(lb2gt_7,1065). happensAt(walking(lb2gt_7),1065).
orientation(lb2gt_5,169,1066). coords(lb2gt_5,206,210,1066).  visible(lb2gt_5,1066). happensAt(walking(lb2gt_5),1066). orientation(lb2gt_7,160,1066). coords(lb2gt_7,79,260,1066).  visible(lb2gt_7,1066). happensAt(walking(lb2gt_7),1066).
#show holdsAt/2.