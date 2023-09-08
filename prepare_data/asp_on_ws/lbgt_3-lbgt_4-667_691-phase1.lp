
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

	orientation(lbgt_3,61,667). coords(lbgt_3,146,162,667).  visible(lbgt_3,667). happensAt(inactive(lbgt_3),667). orientation(lbgt_4,161,667). coords(lbgt_4,130,188,667).  visible(lbgt_4,667). happensAt(active(lbgt_4),667).
orientation(lbgt_3,61,668). coords(lbgt_3,146,162,668).  visible(lbgt_3,668). happensAt(inactive(lbgt_3),668). orientation(lbgt_4,161,668). coords(lbgt_4,130,188,668).  visible(lbgt_4,668). happensAt(active(lbgt_4),668).
orientation(lbgt_3,61,669). coords(lbgt_3,146,162,669).  visible(lbgt_3,669). happensAt(inactive(lbgt_3),669). orientation(lbgt_4,161,669). coords(lbgt_4,130,189,669).  visible(lbgt_4,669). happensAt(active(lbgt_4),669).
orientation(lbgt_3,61,670). coords(lbgt_3,146,162,670).  visible(lbgt_3,670). happensAt(inactive(lbgt_3),670). orientation(lbgt_4,161,670). coords(lbgt_4,130,189,670).  visible(lbgt_4,670). happensAt(active(lbgt_4),670).
orientation(lbgt_3,61,671). coords(lbgt_3,146,162,671).  visible(lbgt_3,671). happensAt(inactive(lbgt_3),671). orientation(lbgt_4,161,671). coords(lbgt_4,130,189,671).  visible(lbgt_4,671). happensAt(active(lbgt_4),671).
orientation(lbgt_3,61,672). coords(lbgt_3,146,162,672).  visible(lbgt_3,672). happensAt(inactive(lbgt_3),672). orientation(lbgt_4,161,672). coords(lbgt_4,130,189,672).  visible(lbgt_4,672). happensAt(active(lbgt_4),672).
orientation(lbgt_3,61,673). coords(lbgt_3,146,162,673).  visible(lbgt_3,673). happensAt(inactive(lbgt_3),673). orientation(lbgt_4,161,673). coords(lbgt_4,130,189,673).  visible(lbgt_4,673). happensAt(active(lbgt_4),673).
orientation(lbgt_3,61,674). coords(lbgt_3,146,162,674).  visible(lbgt_3,674). happensAt(inactive(lbgt_3),674). orientation(lbgt_4,161,674). coords(lbgt_4,130,189,674).  visible(lbgt_4,674). happensAt(active(lbgt_4),674).
orientation(lbgt_3,61,675). coords(lbgt_3,146,162,675).  visible(lbgt_3,675). happensAt(inactive(lbgt_3),675). orientation(lbgt_4,161,675). coords(lbgt_4,130,189,675).  visible(lbgt_4,675). happensAt(active(lbgt_4),675).
orientation(lbgt_3,61,676). coords(lbgt_3,146,162,676).  visible(lbgt_3,676). happensAt(inactive(lbgt_3),676). orientation(lbgt_4,161,676). coords(lbgt_4,130,189,676).  visible(lbgt_4,676). happensAt(active(lbgt_4),676).
orientation(lbgt_3,61,677). coords(lbgt_3,146,162,677).  visible(lbgt_3,677). happensAt(inactive(lbgt_3),677). orientation(lbgt_4,161,677). coords(lbgt_4,130,189,677).  visible(lbgt_4,677). happensAt(active(lbgt_4),677).
orientation(lbgt_3,61,678). coords(lbgt_3,146,162,678).  visible(lbgt_3,678). happensAt(inactive(lbgt_3),678). orientation(lbgt_4,161,678). coords(lbgt_4,130,189,678).  visible(lbgt_4,678). happensAt(active(lbgt_4),678).
orientation(lbgt_3,61,679). coords(lbgt_3,146,162,679).  visible(lbgt_3,679). happensAt(inactive(lbgt_3),679). orientation(lbgt_4,161,679). coords(lbgt_4,130,188,679).  visible(lbgt_4,679). happensAt(active(lbgt_4),679).
orientation(lbgt_3,61,680). coords(lbgt_3,146,162,680).  visible(lbgt_3,680). happensAt(inactive(lbgt_3),680). orientation(lbgt_4,161,680). coords(lbgt_4,130,188,680).  visible(lbgt_4,680). happensAt(active(lbgt_4),680).
orientation(lbgt_3,61,681). coords(lbgt_3,146,162,681).  visible(lbgt_3,681). happensAt(inactive(lbgt_3),681). orientation(lbgt_4,161,681). coords(lbgt_4,129,188,681).  visible(lbgt_4,681). happensAt(active(lbgt_4),681).
orientation(lbgt_3,61,682). coords(lbgt_3,146,162,682).  visible(lbgt_3,682). happensAt(inactive(lbgt_3),682). orientation(lbgt_4,161,682). coords(lbgt_4,129,188,682).  visible(lbgt_4,682). happensAt(active(lbgt_4),682).
orientation(lbgt_3,61,683). coords(lbgt_3,146,162,683).  visible(lbgt_3,683). happensAt(inactive(lbgt_3),683). orientation(lbgt_4,161,683). coords(lbgt_4,129,188,683).  visible(lbgt_4,683). happensAt(walking(lbgt_4),683).
orientation(lbgt_3,61,684). coords(lbgt_3,146,162,684).  visible(lbgt_3,684). happensAt(inactive(lbgt_3),684). orientation(lbgt_4,161,684). coords(lbgt_4,128,187,684).  visible(lbgt_4,684). happensAt(walking(lbgt_4),684).
orientation(lbgt_3,61,685). coords(lbgt_3,146,162,685).  visible(lbgt_3,685). happensAt(inactive(lbgt_3),685). orientation(lbgt_4,161,685). coords(lbgt_4,128,187,685).  visible(lbgt_4,685). happensAt(walking(lbgt_4),685).
orientation(lbgt_3,61,686). coords(lbgt_3,146,162,686).  visible(lbgt_3,686). happensAt(inactive(lbgt_3),686). orientation(lbgt_4,161,686). coords(lbgt_4,128,187,686).  visible(lbgt_4,686). happensAt(walking(lbgt_4),686).
orientation(lbgt_3,61,687). coords(lbgt_3,146,162,687).  visible(lbgt_3,687). happensAt(inactive(lbgt_3),687). orientation(lbgt_4,161,687). coords(lbgt_4,128,186,687).  visible(lbgt_4,687). happensAt(walking(lbgt_4),687).
orientation(lbgt_3,61,688). coords(lbgt_3,146,162,688).  visible(lbgt_3,688). happensAt(inactive(lbgt_3),688). orientation(lbgt_4,161,688). coords(lbgt_4,126,185,688).  visible(lbgt_4,688). happensAt(walking(lbgt_4),688).
orientation(lbgt_3,61,689). coords(lbgt_3,146,162,689).  visible(lbgt_3,689). happensAt(inactive(lbgt_3),689). orientation(lbgt_4,160,689). coords(lbgt_4,125,185,689).  visible(lbgt_4,689). happensAt(walking(lbgt_4),689).
orientation(lbgt_3,61,690). coords(lbgt_3,146,162,690).  visible(lbgt_3,690). happensAt(inactive(lbgt_3),690). orientation(lbgt_4,158,690). coords(lbgt_4,123,185,690).  visible(lbgt_4,690). happensAt(walking(lbgt_4),690).
orientation(lbgt_3,61,691). coords(lbgt_3,146,162,691).  visible(lbgt_3,691). happensAt(inactive(lbgt_3),691). orientation(lbgt_4,158,691). coords(lbgt_4,122,184,691).  visible(lbgt_4,691). happensAt(walking(lbgt_4),691).
#show holdsAt/2.