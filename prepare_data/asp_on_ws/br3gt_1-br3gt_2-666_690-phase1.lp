
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

	orientation(br3gt_1,131,666). coords(br3gt_1,185,46,666).  visible(br3gt_1,666). happensAt(walking(br3gt_1),666). orientation(br3gt_2,148,666). coords(br3gt_2,8,68,666).  visible(br3gt_2,666). happensAt(walking(br3gt_2),666).
orientation(br3gt_1,131,667). coords(br3gt_1,186,46,667).  visible(br3gt_1,667). happensAt(walking(br3gt_1),667). orientation(br3gt_2,148,667). coords(br3gt_2,9,67,667).  visible(br3gt_2,667). happensAt(walking(br3gt_2),667).
orientation(br3gt_1,131,668). coords(br3gt_1,186,46,668).  visible(br3gt_1,668). happensAt(walking(br3gt_1),668). orientation(br3gt_2,148,668). coords(br3gt_2,9,66,668).  visible(br3gt_2,668). happensAt(walking(br3gt_2),668).
orientation(br3gt_1,131,669). coords(br3gt_1,186,46,669).  visible(br3gt_1,669). happensAt(walking(br3gt_1),669). orientation(br3gt_2,148,669). coords(br3gt_2,9,65,669).  visible(br3gt_2,669). happensAt(walking(br3gt_2),669).
orientation(br3gt_1,131,670). coords(br3gt_1,186,46,670).  visible(br3gt_1,670). happensAt(walking(br3gt_1),670). orientation(br3gt_2,148,670). coords(br3gt_2,9,64,670).  visible(br3gt_2,670). happensAt(walking(br3gt_2),670).
orientation(br3gt_1,131,671). coords(br3gt_1,186,46,671).  visible(br3gt_1,671). happensAt(walking(br3gt_1),671). orientation(br3gt_2,148,671). coords(br3gt_2,9,63,671).  visible(br3gt_2,671). happensAt(walking(br3gt_2),671).
orientation(br3gt_1,131,672). coords(br3gt_1,186,47,672).  visible(br3gt_1,672). happensAt(walking(br3gt_1),672). orientation(br3gt_2,148,672). coords(br3gt_2,9,63,672).  visible(br3gt_2,672). happensAt(walking(br3gt_2),672).
orientation(br3gt_1,131,673). coords(br3gt_1,186,47,673).  visible(br3gt_1,673). happensAt(walking(br3gt_1),673). orientation(br3gt_2,148,673). coords(br3gt_2,9,62,673).  visible(br3gt_2,673). happensAt(walking(br3gt_2),673).
orientation(br3gt_1,131,674). coords(br3gt_1,186,47,674).  visible(br3gt_1,674). happensAt(walking(br3gt_1),674). orientation(br3gt_2,148,674). coords(br3gt_2,10,61,674).  visible(br3gt_2,674). happensAt(walking(br3gt_2),674).
orientation(br3gt_1,131,675). coords(br3gt_1,186,47,675).  visible(br3gt_1,675). happensAt(walking(br3gt_1),675). orientation(br3gt_2,148,675). coords(br3gt_2,11,61,675).  visible(br3gt_2,675). happensAt(walking(br3gt_2),675).
orientation(br3gt_1,131,676). coords(br3gt_1,186,47,676).  visible(br3gt_1,676). happensAt(walking(br3gt_1),676). orientation(br3gt_2,141,676). coords(br3gt_2,11,60,676).  visible(br3gt_2,676). happensAt(walking(br3gt_2),676).
orientation(br3gt_1,131,677). coords(br3gt_1,186,47,677).  visible(br3gt_1,677). happensAt(walking(br3gt_1),677). orientation(br3gt_2,141,677). coords(br3gt_2,11,60,677).  visible(br3gt_2,677). happensAt(walking(br3gt_2),677).
orientation(br3gt_1,131,678). coords(br3gt_1,186,47,678).  visible(br3gt_1,678). happensAt(walking(br3gt_1),678). orientation(br3gt_2,138,678). coords(br3gt_2,12,59,678).  visible(br3gt_2,678). happensAt(walking(br3gt_2),678).
orientation(br3gt_1,131,679). coords(br3gt_1,186,47,679).  visible(br3gt_1,679). happensAt(walking(br3gt_1),679). orientation(br3gt_2,135,679). coords(br3gt_2,11,57,679).  visible(br3gt_2,679). happensAt(walking(br3gt_2),679).
orientation(br3gt_1,136,680). coords(br3gt_1,188,47,680).  visible(br3gt_1,680). happensAt(walking(br3gt_1),680). orientation(br3gt_2,135,680). coords(br3gt_2,13,57,680).  visible(br3gt_2,680). happensAt(walking(br3gt_2),680).
orientation(br3gt_1,136,681). coords(br3gt_1,190,48,681).  visible(br3gt_1,681). happensAt(walking(br3gt_1),681). orientation(br3gt_2,138,681). coords(br3gt_2,13,56,681).  visible(br3gt_2,681). happensAt(walking(br3gt_2),681).
orientation(br3gt_1,136,682). coords(br3gt_1,191,48,682).  visible(br3gt_1,682). happensAt(walking(br3gt_1),682). orientation(br3gt_2,141,682). coords(br3gt_2,12,57,682).  visible(br3gt_2,682). happensAt(walking(br3gt_2),682).
orientation(br3gt_1,136,683). coords(br3gt_1,192,48,683).  visible(br3gt_1,683). happensAt(walking(br3gt_1),683). orientation(br3gt_2,144,683). coords(br3gt_2,11,55,683).  visible(br3gt_2,683). happensAt(walking(br3gt_2),683).
orientation(br3gt_1,138,684). coords(br3gt_1,192,50,684).  visible(br3gt_1,684). happensAt(walking(br3gt_1),684). orientation(br3gt_2,144,684). coords(br3gt_2,11,54,684).  visible(br3gt_2,684). happensAt(walking(br3gt_2),684).
orientation(br3gt_1,138,685). coords(br3gt_1,192,50,685).  visible(br3gt_1,685). happensAt(walking(br3gt_1),685). orientation(br3gt_2,144,685). coords(br3gt_2,11,55,685).  visible(br3gt_2,685). happensAt(walking(br3gt_2),685).
orientation(br3gt_1,138,686). coords(br3gt_1,192,51,686).  visible(br3gt_1,686). happensAt(walking(br3gt_1),686). orientation(br3gt_2,144,686). coords(br3gt_2,11,54,686).  visible(br3gt_2,686). happensAt(walking(br3gt_2),686).
orientation(br3gt_1,138,687). coords(br3gt_1,193,51,687).  visible(br3gt_1,687). happensAt(walking(br3gt_1),687). orientation(br3gt_2,144,687). coords(br3gt_2,11,53,687).  visible(br3gt_2,687). happensAt(walking(br3gt_2),687).
orientation(br3gt_1,138,688). coords(br3gt_1,193,52,688).  visible(br3gt_1,688). happensAt(walking(br3gt_1),688). orientation(br3gt_2,144,688). coords(br3gt_2,11,52,688).  visible(br3gt_2,688). happensAt(walking(br3gt_2),688).
orientation(br3gt_1,138,689). coords(br3gt_1,193,52,689).  visible(br3gt_1,689). happensAt(walking(br3gt_1),689). orientation(br3gt_2,144,689). coords(br3gt_2,11,52,689).  happensAt(occluded(br3gt_2),689). happensAt(walking(br3gt_2),689).
orientation(br3gt_1,138,690). coords(br3gt_1,194,52,690).  visible(br3gt_1,690). happensAt(walking(br3gt_1),690). orientation(br3gt_2,144,690). coords(br3gt_2,11,52,690).  happensAt(occluded(br3gt_2),690). happensAt(walking(br3gt_2),690).
#show holdsAt/2.