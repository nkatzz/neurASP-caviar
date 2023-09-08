
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

	orientation(lbbcgt_5,154,1017). coords(lbbcgt_5,178,117,1017).  visible(lbbcgt_5,1017). happensAt(active(lbbcgt_5),1017). orientation(lbbcgt_6,151,1017). coords(lbbcgt_6,62,94,1017).  visible(lbbcgt_6,1017). happensAt(walking(lbbcgt_6),1017).
orientation(lbbcgt_5,154,1018). coords(lbbcgt_5,179,117,1018).  visible(lbbcgt_5,1018). happensAt(active(lbbcgt_5),1018). orientation(lbbcgt_6,151,1018). coords(lbbcgt_6,62,94,1018).  visible(lbbcgt_6,1018). happensAt(walking(lbbcgt_6),1018).
orientation(lbbcgt_5,154,1019). coords(lbbcgt_5,179,117,1019).  visible(lbbcgt_5,1019). happensAt(active(lbbcgt_5),1019). orientation(lbbcgt_6,151,1019). coords(lbbcgt_6,62,94,1019).  visible(lbbcgt_6,1019). happensAt(walking(lbbcgt_6),1019).
orientation(lbbcgt_5,154,1020). coords(lbbcgt_5,180,117,1020).  visible(lbbcgt_5,1020). happensAt(active(lbbcgt_5),1020). orientation(lbbcgt_6,151,1020). coords(lbbcgt_6,62,94,1020).  visible(lbbcgt_6,1020). happensAt(walking(lbbcgt_6),1020).
orientation(lbbcgt_5,154,1021). coords(lbbcgt_5,181,117,1021).  visible(lbbcgt_5,1021). happensAt(active(lbbcgt_5),1021). orientation(lbbcgt_6,151,1021). coords(lbbcgt_6,62,94,1021).  visible(lbbcgt_6,1021). happensAt(walking(lbbcgt_6),1021).
orientation(lbbcgt_5,154,1022). coords(lbbcgt_5,183,117,1022).  visible(lbbcgt_5,1022). happensAt(active(lbbcgt_5),1022). orientation(lbbcgt_6,151,1022). coords(lbbcgt_6,62,94,1022).  visible(lbbcgt_6,1022). happensAt(walking(lbbcgt_6),1022).
orientation(lbbcgt_5,154,1023). coords(lbbcgt_5,184,117,1023).  visible(lbbcgt_5,1023). happensAt(active(lbbcgt_5),1023). orientation(lbbcgt_6,151,1023). coords(lbbcgt_6,62,94,1023).  visible(lbbcgt_6,1023). happensAt(walking(lbbcgt_6),1023).
orientation(lbbcgt_5,154,1024). coords(lbbcgt_5,185,117,1024).  visible(lbbcgt_5,1024). happensAt(active(lbbcgt_5),1024). orientation(lbbcgt_6,151,1024). coords(lbbcgt_6,62,94,1024).  visible(lbbcgt_6,1024). happensAt(walking(lbbcgt_6),1024).
orientation(lbbcgt_5,154,1025). coords(lbbcgt_5,185,117,1025).  visible(lbbcgt_5,1025). happensAt(active(lbbcgt_5),1025). orientation(lbbcgt_6,151,1025). coords(lbbcgt_6,62,94,1025).  visible(lbbcgt_6,1025). happensAt(walking(lbbcgt_6),1025).
orientation(lbbcgt_5,154,1026). coords(lbbcgt_5,185,117,1026).  visible(lbbcgt_5,1026). happensAt(active(lbbcgt_5),1026). orientation(lbbcgt_6,151,1026). coords(lbbcgt_6,62,94,1026).  visible(lbbcgt_6,1026). happensAt(walking(lbbcgt_6),1026).
orientation(lbbcgt_5,154,1027). coords(lbbcgt_5,185,117,1027).  visible(lbbcgt_5,1027). happensAt(active(lbbcgt_5),1027). orientation(lbbcgt_6,151,1027). coords(lbbcgt_6,62,94,1027).  visible(lbbcgt_6,1027). happensAt(walking(lbbcgt_6),1027).
orientation(lbbcgt_5,154,1028). coords(lbbcgt_5,185,117,1028).  visible(lbbcgt_5,1028). happensAt(active(lbbcgt_5),1028). orientation(lbbcgt_6,151,1028). coords(lbbcgt_6,62,95,1028).  visible(lbbcgt_6,1028). happensAt(walking(lbbcgt_6),1028).
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
#show holdsAt/2.