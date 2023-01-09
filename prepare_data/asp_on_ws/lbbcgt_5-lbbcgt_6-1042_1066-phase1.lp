
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

	orientation(lbbcgt_5,146,1042). coords(lbbcgt_5,189,116,1042).  visible(lbbcgt_5,1042). happensAt(active(lbbcgt_5),1042). orientation(lbbcgt_6,161,1042). coords(lbbcgt_6,56,98,1042).  visible(lbbcgt_6,1042). happensAt(walking(lbbcgt_6),1042).
orientation(lbbcgt_5,145,1043). coords(lbbcgt_5,189,116,1043).  visible(lbbcgt_5,1043). happensAt(active(lbbcgt_5),1043). orientation(lbbcgt_6,163,1043). coords(lbbcgt_6,55,98,1043).  visible(lbbcgt_6,1043). happensAt(walking(lbbcgt_6),1043).
orientation(lbbcgt_5,147,1044). coords(lbbcgt_5,189,116,1044).  visible(lbbcgt_5,1044). happensAt(active(lbbcgt_5),1044). orientation(lbbcgt_6,168,1044). coords(lbbcgt_6,55,98,1044).  visible(lbbcgt_6,1044). happensAt(walking(lbbcgt_6),1044).
orientation(lbbcgt_5,147,1045). coords(lbbcgt_5,189,116,1045).  visible(lbbcgt_5,1045). happensAt(active(lbbcgt_5),1045). orientation(lbbcgt_6,162,1045). coords(lbbcgt_6,54,100,1045).  visible(lbbcgt_6,1045). happensAt(walking(lbbcgt_6),1045).
orientation(lbbcgt_5,142,1046). coords(lbbcgt_5,190,116,1046).  visible(lbbcgt_5,1046). happensAt(active(lbbcgt_5),1046). orientation(lbbcgt_6,159,1046). coords(lbbcgt_6,55,101,1046).  visible(lbbcgt_6,1046). happensAt(walking(lbbcgt_6),1046).
orientation(lbbcgt_5,142,1047). coords(lbbcgt_5,190,116,1047).  visible(lbbcgt_5,1047). happensAt(active(lbbcgt_5),1047). orientation(lbbcgt_6,162,1047). coords(lbbcgt_6,54,102,1047).  visible(lbbcgt_6,1047). happensAt(walking(lbbcgt_6),1047).
orientation(lbbcgt_5,142,1048). coords(lbbcgt_5,190,116,1048).  visible(lbbcgt_5,1048). happensAt(active(lbbcgt_5),1048). orientation(lbbcgt_6,160,1048). coords(lbbcgt_6,52,102,1048).  visible(lbbcgt_6,1048). happensAt(walking(lbbcgt_6),1048).
orientation(lbbcgt_5,142,1049). coords(lbbcgt_5,191,116,1049).  visible(lbbcgt_5,1049). happensAt(active(lbbcgt_5),1049). orientation(lbbcgt_6,151,1049). coords(lbbcgt_6,52,103,1049).  visible(lbbcgt_6,1049). happensAt(walking(lbbcgt_6),1049).
orientation(lbbcgt_5,142,1050). coords(lbbcgt_5,191,116,1050).  visible(lbbcgt_5,1050). happensAt(active(lbbcgt_5),1050). orientation(lbbcgt_6,157,1050). coords(lbbcgt_6,52,103,1050).  visible(lbbcgt_6,1050). happensAt(walking(lbbcgt_6),1050).
orientation(lbbcgt_5,138,1051). coords(lbbcgt_5,191,116,1051).  visible(lbbcgt_5,1051). happensAt(active(lbbcgt_5),1051). orientation(lbbcgt_6,164,1051). coords(lbbcgt_6,52,103,1051).  visible(lbbcgt_6,1051). happensAt(walking(lbbcgt_6),1051).
orientation(lbbcgt_5,138,1052). coords(lbbcgt_5,191,116,1052).  visible(lbbcgt_5,1052). happensAt(active(lbbcgt_5),1052). orientation(lbbcgt_6,161,1052). coords(lbbcgt_6,52,103,1052).  visible(lbbcgt_6,1052). happensAt(walking(lbbcgt_6),1052).
orientation(lbbcgt_5,138,1053). coords(lbbcgt_5,191,116,1053).  visible(lbbcgt_5,1053). happensAt(active(lbbcgt_5),1053). orientation(lbbcgt_6,160,1053). coords(lbbcgt_6,51,104,1053).  visible(lbbcgt_6,1053). happensAt(walking(lbbcgt_6),1053).
orientation(lbbcgt_5,138,1054). coords(lbbcgt_5,191,116,1054).  visible(lbbcgt_5,1054). happensAt(active(lbbcgt_5),1054). orientation(lbbcgt_6,163,1054). coords(lbbcgt_6,51,105,1054).  visible(lbbcgt_6,1054). happensAt(walking(lbbcgt_6),1054).
orientation(lbbcgt_5,138,1055). coords(lbbcgt_5,191,116,1055).  visible(lbbcgt_5,1055). happensAt(active(lbbcgt_5),1055). orientation(lbbcgt_6,160,1055). coords(lbbcgt_6,52,105,1055).  visible(lbbcgt_6,1055). happensAt(walking(lbbcgt_6),1055).
orientation(lbbcgt_5,138,1056). coords(lbbcgt_5,191,117,1056).  visible(lbbcgt_5,1056). happensAt(active(lbbcgt_5),1056). orientation(lbbcgt_6,175,1056). coords(lbbcgt_6,47,105,1056).  visible(lbbcgt_6,1056). happensAt(walking(lbbcgt_6),1056).
orientation(lbbcgt_5,138,1057). coords(lbbcgt_5,191,118,1057).  visible(lbbcgt_5,1057). happensAt(active(lbbcgt_5),1057). orientation(lbbcgt_6,172,1057). coords(lbbcgt_6,46,106,1057).  visible(lbbcgt_6,1057). happensAt(walking(lbbcgt_6),1057).
orientation(lbbcgt_5,138,1058). coords(lbbcgt_5,191,118,1058).  visible(lbbcgt_5,1058). happensAt(active(lbbcgt_5),1058). orientation(lbbcgt_6,172,1058). coords(lbbcgt_6,46,106,1058).  visible(lbbcgt_6,1058). happensAt(walking(lbbcgt_6),1058).
orientation(lbbcgt_5,138,1059). coords(lbbcgt_5,191,118,1059).  visible(lbbcgt_5,1059). happensAt(active(lbbcgt_5),1059). orientation(lbbcgt_6,172,1059). coords(lbbcgt_6,46,107,1059).  visible(lbbcgt_6,1059). happensAt(walking(lbbcgt_6),1059).
orientation(lbbcgt_5,138,1060). coords(lbbcgt_5,191,118,1060).  visible(lbbcgt_5,1060). happensAt(active(lbbcgt_5),1060). orientation(lbbcgt_6,172,1060). coords(lbbcgt_6,46,107,1060).  visible(lbbcgt_6,1060). happensAt(walking(lbbcgt_6),1060).
orientation(lbbcgt_5,138,1061). coords(lbbcgt_5,191,118,1061).  visible(lbbcgt_5,1061). happensAt(active(lbbcgt_5),1061). orientation(lbbcgt_6,7,1061). coords(lbbcgt_6,47,109,1061).  visible(lbbcgt_6,1061). happensAt(walking(lbbcgt_6),1061).
orientation(lbbcgt_5,138,1062). coords(lbbcgt_5,190,118,1062).  visible(lbbcgt_5,1062). happensAt(active(lbbcgt_5),1062). orientation(lbbcgt_6,11,1062). coords(lbbcgt_6,47,109,1062).  visible(lbbcgt_6,1062). happensAt(walking(lbbcgt_6),1062).
orientation(lbbcgt_5,138,1063). coords(lbbcgt_5,190,118,1063).  visible(lbbcgt_5,1063). happensAt(active(lbbcgt_5),1063). orientation(lbbcgt_6,18,1063). coords(lbbcgt_6,47,110,1063).  visible(lbbcgt_6,1063). happensAt(walking(lbbcgt_6),1063).
orientation(lbbcgt_5,138,1064). coords(lbbcgt_5,190,118,1064).  visible(lbbcgt_5,1064). happensAt(active(lbbcgt_5),1064). orientation(lbbcgt_6,25,1064). coords(lbbcgt_6,47,111,1064).  visible(lbbcgt_6,1064). happensAt(walking(lbbcgt_6),1064).
orientation(lbbcgt_5,138,1065). coords(lbbcgt_5,190,118,1065).  visible(lbbcgt_5,1065). happensAt(active(lbbcgt_5),1065). orientation(lbbcgt_6,25,1065). coords(lbbcgt_6,47,111,1065).  visible(lbbcgt_6,1065). happensAt(walking(lbbcgt_6),1065).
orientation(lbbcgt_5,138,1066). coords(lbbcgt_5,190,118,1066).  happensAt(disappear(lbbcgt_5),1066). happensAt(active(lbbcgt_5),1066). orientation(lbbcgt_6,16,1066). coords(lbbcgt_6,47,111,1066).  happensAt(disappear(lbbcgt_6),1066). happensAt(walking(lbbcgt_6),1066).
#show holdsAt/2.