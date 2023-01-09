
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

	orientation(lb2gt_5,152,1030). coords(lb2gt_5,164,158,1030).  visible(lb2gt_5,1030). happensAt(walking(lb2gt_5),1030). orientation(lb2gt_7,0,1030). coords(lb2gt_7,89,268,1030).  visible(lb2gt_7,1030). happensAt(walking(lb2gt_7),1030).
orientation(lb2gt_5,152,1031). coords(lb2gt_5,166,160,1031).  visible(lb2gt_5,1031). happensAt(walking(lb2gt_5),1031). orientation(lb2gt_7,0,1031). coords(lb2gt_7,89,268,1031).  visible(lb2gt_7,1031). happensAt(walking(lb2gt_7),1031).
orientation(lb2gt_5,152,1032). coords(lb2gt_5,168,161,1032).  visible(lb2gt_5,1032). happensAt(walking(lb2gt_5),1032). orientation(lb2gt_7,0,1032). coords(lb2gt_7,89,268,1032).  visible(lb2gt_7,1032). happensAt(walking(lb2gt_7),1032).
orientation(lb2gt_5,152,1033). coords(lb2gt_5,169,162,1033).  visible(lb2gt_5,1033). happensAt(walking(lb2gt_5),1033). orientation(lb2gt_7,0,1033). coords(lb2gt_7,89,267,1033).  visible(lb2gt_7,1033). happensAt(walking(lb2gt_7),1033).
orientation(lb2gt_5,152,1034). coords(lb2gt_5,170,163,1034).  visible(lb2gt_5,1034). happensAt(walking(lb2gt_5),1034). orientation(lb2gt_7,0,1034). coords(lb2gt_7,88,267,1034).  visible(lb2gt_7,1034). happensAt(walking(lb2gt_7),1034).
orientation(lb2gt_5,152,1035). coords(lb2gt_5,171,164,1035).  visible(lb2gt_5,1035). happensAt(walking(lb2gt_5),1035). orientation(lb2gt_7,0,1035). coords(lb2gt_7,88,267,1035).  visible(lb2gt_7,1035). happensAt(walking(lb2gt_7),1035).
orientation(lb2gt_5,152,1036). coords(lb2gt_5,172,164,1036).  visible(lb2gt_5,1036). happensAt(walking(lb2gt_5),1036). orientation(lb2gt_7,0,1036). coords(lb2gt_7,88,267,1036).  visible(lb2gt_7,1036). happensAt(walking(lb2gt_7),1036).
orientation(lb2gt_5,152,1037). coords(lb2gt_5,172,165,1037).  visible(lb2gt_5,1037). happensAt(walking(lb2gt_5),1037). orientation(lb2gt_7,0,1037). coords(lb2gt_7,88,268,1037).  visible(lb2gt_7,1037). happensAt(walking(lb2gt_7),1037).
orientation(lb2gt_5,152,1038). coords(lb2gt_5,173,166,1038).  visible(lb2gt_5,1038). happensAt(walking(lb2gt_5),1038). orientation(lb2gt_7,0,1038). coords(lb2gt_7,88,267,1038).  visible(lb2gt_7,1038). happensAt(walking(lb2gt_7),1038).
orientation(lb2gt_5,152,1039). coords(lb2gt_5,173,166,1039).  visible(lb2gt_5,1039). happensAt(walking(lb2gt_5),1039). orientation(lb2gt_7,0,1039). coords(lb2gt_7,88,267,1039).  visible(lb2gt_7,1039). happensAt(walking(lb2gt_7),1039).
orientation(lb2gt_5,152,1040). coords(lb2gt_5,173,167,1040).  visible(lb2gt_5,1040). happensAt(walking(lb2gt_5),1040). orientation(lb2gt_7,0,1040). coords(lb2gt_7,88,266,1040).  visible(lb2gt_7,1040). happensAt(walking(lb2gt_7),1040).
orientation(lb2gt_5,152,1041). coords(lb2gt_5,175,168,1041).  visible(lb2gt_5,1041). happensAt(walking(lb2gt_5),1041). orientation(lb2gt_7,0,1041). coords(lb2gt_7,88,266,1041).  visible(lb2gt_7,1041). happensAt(walking(lb2gt_7),1041).
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
#show holdsAt/2.