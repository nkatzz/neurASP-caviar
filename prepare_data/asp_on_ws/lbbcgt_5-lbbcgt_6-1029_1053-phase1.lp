
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

	orientation(lbbcgt_5,154,1029). coords(lbbcgt_5,185,117,1029).  visible(lbbcgt_5,1029). happensAt(active(lbbcgt_5),1029). orientation(lbbcgt_6,151,1029). coords(lbbcgt_6,62,95,1029).  visible(lbbcgt_6,1029). happensAt(walking(lbbcgt_6),1029).
orientation(lbbcgt_5,154,1030). coords(lbbcgt_5,185,117,1030).  visible(lbbcgt_5,1030). happensAt(active(lbbcgt_5),1030). orientation(lbbcgt_6,151,1030). coords(lbbcgt_6,62,95,1030).  visible(lbbcgt_6,1030). happensAt(walking(lbbcgt_6),1030).
orientation(lbbcgt_5,154,1031). coords(lbbcgt_5,185,117,1031).  visible(lbbcgt_5,1031). happensAt(active(lbbcgt_5),1031). orientation(lbbcgt_6,151,1031). coords(lbbcgt_6,61,96,1031).  visible(lbbcgt_6,1031). happensAt(walking(lbbcgt_6),1031).
orientation(lbbcgt_5,154,1032). coords(lbbcgt_5,185,117,1032).  visible(lbbcgt_5,1032). happensAt(active(lbbcgt_5),1032). orientation(lbbcgt_6,151,1032). coords(lbbcgt_6,61,96,1032).  visible(lbbcgt_6,1032). happensAt(walking(lbbcgt_6),1032).
orientation(lbbcgt_5,154,1033). coords(lbbcgt_5,185,117,1033).  visible(lbbcgt_5,1033). happensAt(active(lbbcgt_5),1033). orientation(lbbcgt_6,151,1033). coords(lbbcgt_6,61,96,1033).  visible(lbbcgt_6,1033). happensAt(walking(lbbcgt_6),1033).
orientation(lbbcgt_5,154,1034). coords(lbbcgt_5,185,117,1034).  visible(lbbcgt_5,1034). happensAt(active(lbbcgt_5),1034). orientation(lbbcgt_6,151,1034). coords(lbbcgt_6,60,96,1034).  visible(lbbcgt_6,1034). happensAt(walking(lbbcgt_6),1034).
orientation(lbbcgt_5,154,1035). coords(lbbcgt_5,186,117,1035).  visible(lbbcgt_5,1035). happensAt(active(lbbcgt_5),1035). orientation(lbbcgt_6,151,1035). coords(lbbcgt_6,60,96,1035).  visible(lbbcgt_6,1035). happensAt(walking(lbbcgt_6),1035).
orientation(lbbcgt_5,154,1036). coords(lbbcgt_5,187,117,1036).  visible(lbbcgt_5,1036). happensAt(active(lbbcgt_5),1036). orientation(lbbcgt_6,151,1036). coords(lbbcgt_6,60,97,1036).  visible(lbbcgt_6,1036). happensAt(walking(lbbcgt_6),1036).
orientation(lbbcgt_5,154,1037). coords(lbbcgt_5,187,116,1037).  visible(lbbcgt_5,1037). happensAt(active(lbbcgt_5),1037). orientation(lbbcgt_6,151,1037). coords(lbbcgt_6,59,97,1037).  visible(lbbcgt_6,1037). happensAt(walking(lbbcgt_6),1037).
orientation(lbbcgt_5,154,1038). coords(lbbcgt_5,187,116,1038).  visible(lbbcgt_5,1038). happensAt(active(lbbcgt_5),1038). orientation(lbbcgt_6,151,1038). coords(lbbcgt_6,59,97,1038).  visible(lbbcgt_6,1038). happensAt(walking(lbbcgt_6),1038).
orientation(lbbcgt_5,154,1039). coords(lbbcgt_5,188,115,1039).  visible(lbbcgt_5,1039). happensAt(active(lbbcgt_5),1039). orientation(lbbcgt_6,168,1039). coords(lbbcgt_6,58,97,1039).  visible(lbbcgt_6,1039). happensAt(walking(lbbcgt_6),1039).
orientation(lbbcgt_5,150,1040). coords(lbbcgt_5,188,115,1040).  visible(lbbcgt_5,1040). happensAt(active(lbbcgt_5),1040). orientation(lbbcgt_6,163,1040). coords(lbbcgt_6,57,97,1040).  visible(lbbcgt_6,1040). happensAt(walking(lbbcgt_6),1040).
orientation(lbbcgt_5,150,1041). coords(lbbcgt_5,189,116,1041).  visible(lbbcgt_5,1041). happensAt(active(lbbcgt_5),1041). orientation(lbbcgt_6,165,1041). coords(lbbcgt_6,56,98,1041).  visible(lbbcgt_6,1041). happensAt(walking(lbbcgt_6),1041).
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
#show holdsAt/2.