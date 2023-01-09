
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

	orientation(bww1gt_1,173,667). coords(bww1gt_1,102,257,667).  visible(bww1gt_1,667). happensAt(inactive(bww1gt_1),667). orientation(bww1gt_2,3,667). coords(bww1gt_2,211,258,667).  visible(bww1gt_2,667). happensAt(walking(bww1gt_2),667).
orientation(bww1gt_1,173,668). coords(bww1gt_1,102,257,668).  visible(bww1gt_1,668). happensAt(inactive(bww1gt_1),668). orientation(bww1gt_2,2,668). coords(bww1gt_2,214,260,668).  visible(bww1gt_2,668). happensAt(walking(bww1gt_2),668).
orientation(bww1gt_1,173,669). coords(bww1gt_1,102,257,669).  visible(bww1gt_1,669). happensAt(inactive(bww1gt_1),669). orientation(bww1gt_2,10,669). coords(bww1gt_2,215,262,669).  visible(bww1gt_2,669). happensAt(walking(bww1gt_2),669).
orientation(bww1gt_1,173,670). coords(bww1gt_1,102,257,670).  visible(bww1gt_1,670). happensAt(inactive(bww1gt_1),670). orientation(bww1gt_2,7,670). coords(bww1gt_2,216,263,670).  visible(bww1gt_2,670). happensAt(walking(bww1gt_2),670).
orientation(bww1gt_1,173,671). coords(bww1gt_1,102,257,671).  visible(bww1gt_1,671). happensAt(inactive(bww1gt_1),671). orientation(bww1gt_2,4,671). coords(bww1gt_2,217,264,671).  visible(bww1gt_2,671). happensAt(walking(bww1gt_2),671).
orientation(bww1gt_1,173,672). coords(bww1gt_1,102,257,672).  visible(bww1gt_1,672). happensAt(inactive(bww1gt_1),672). orientation(bww1gt_2,10,672). coords(bww1gt_2,218,265,672).  visible(bww1gt_2,672). happensAt(walking(bww1gt_2),672).
orientation(bww1gt_1,173,673). coords(bww1gt_1,102,257,673).  visible(bww1gt_1,673). happensAt(inactive(bww1gt_1),673). orientation(bww1gt_2,14,673). coords(bww1gt_2,219,267,673).  visible(bww1gt_2,673). happensAt(walking(bww1gt_2),673).
orientation(bww1gt_1,173,674). coords(bww1gt_1,102,257,674).  visible(bww1gt_1,674). happensAt(inactive(bww1gt_1),674). orientation(bww1gt_2,8,674). coords(bww1gt_2,219,268,674).  visible(bww1gt_2,674). happensAt(walking(bww1gt_2),674).
orientation(bww1gt_1,173,675). coords(bww1gt_1,102,257,675).  visible(bww1gt_1,675). happensAt(inactive(bww1gt_1),675). orientation(bww1gt_2,12,675). coords(bww1gt_2,220,269,675).  visible(bww1gt_2,675). happensAt(walking(bww1gt_2),675).
orientation(bww1gt_1,173,676). coords(bww1gt_1,102,257,676).  visible(bww1gt_1,676). happensAt(inactive(bww1gt_1),676). orientation(bww1gt_2,9,676). coords(bww1gt_2,220,271,676).  visible(bww1gt_2,676). happensAt(walking(bww1gt_2),676).
orientation(bww1gt_1,173,677). coords(bww1gt_1,102,257,677).  visible(bww1gt_1,677). happensAt(inactive(bww1gt_1),677). orientation(bww1gt_2,15,677). coords(bww1gt_2,220,273,677).  visible(bww1gt_2,677). happensAt(walking(bww1gt_2),677).
orientation(bww1gt_1,173,678). coords(bww1gt_1,102,257,678).  visible(bww1gt_1,678). happensAt(inactive(bww1gt_1),678). orientation(bww1gt_2,13,678). coords(bww1gt_2,221,274,678).  visible(bww1gt_2,678). happensAt(walking(bww1gt_2),678).
orientation(bww1gt_1,173,679). coords(bww1gt_1,102,257,679).  visible(bww1gt_1,679). happensAt(inactive(bww1gt_1),679). orientation(bww1gt_2,16,679). coords(bww1gt_2,221,276,679).  visible(bww1gt_2,679). happensAt(walking(bww1gt_2),679).
orientation(bww1gt_1,173,680). coords(bww1gt_1,102,257,680).  visible(bww1gt_1,680). happensAt(inactive(bww1gt_1),680). orientation(bww1gt_2,16,680). coords(bww1gt_2,221,278,680).  visible(bww1gt_2,680). happensAt(walking(bww1gt_2),680).
orientation(bww1gt_1,173,681). coords(bww1gt_1,102,257,681).  visible(bww1gt_1,681). happensAt(inactive(bww1gt_1),681). orientation(bww1gt_2,13,681). coords(bww1gt_2,221,279,681).  visible(bww1gt_2,681). happensAt(walking(bww1gt_2),681).
orientation(bww1gt_1,173,682). coords(bww1gt_1,102,257,682).  visible(bww1gt_1,682). happensAt(inactive(bww1gt_1),682). orientation(bww1gt_2,12,682). coords(bww1gt_2,222,279,682).  visible(bww1gt_2,682). happensAt(walking(bww1gt_2),682).
orientation(bww1gt_1,173,683). coords(bww1gt_1,102,257,683).  visible(bww1gt_1,683). happensAt(inactive(bww1gt_1),683). orientation(bww1gt_2,26,683). coords(bww1gt_2,222,279,683).  visible(bww1gt_2,683). happensAt(walking(bww1gt_2),683).
orientation(bww1gt_1,173,684). coords(bww1gt_1,102,257,684).  visible(bww1gt_1,684). happensAt(inactive(bww1gt_1),684). orientation(bww1gt_2,25,684). coords(bww1gt_2,228,279,684).  visible(bww1gt_2,684). happensAt(walking(bww1gt_2),684).
orientation(bww1gt_1,173,685). coords(bww1gt_1,102,257,685).  visible(bww1gt_1,685). happensAt(inactive(bww1gt_1),685). orientation(bww1gt_2,25,685). coords(bww1gt_2,229,279,685).  visible(bww1gt_2,685). happensAt(walking(bww1gt_2),685).
orientation(bww1gt_1,173,686). coords(bww1gt_1,102,257,686).  visible(bww1gt_1,686). happensAt(inactive(bww1gt_1),686). orientation(bww1gt_2,47,686). coords(bww1gt_2,234,279,686).  visible(bww1gt_2,686). happensAt(walking(bww1gt_2),686).
orientation(bww1gt_1,173,687). coords(bww1gt_1,102,257,687).  visible(bww1gt_1,687). happensAt(inactive(bww1gt_1),687). orientation(bww1gt_2,47,687). coords(bww1gt_2,236,279,687).  visible(bww1gt_2,687). happensAt(walking(bww1gt_2),687).
orientation(bww1gt_1,173,688). coords(bww1gt_1,102,257,688).  visible(bww1gt_1,688). happensAt(inactive(bww1gt_1),688). orientation(bww1gt_2,47,688). coords(bww1gt_2,236,279,688).  visible(bww1gt_2,688). happensAt(walking(bww1gt_2),688).
orientation(bww1gt_1,173,689). coords(bww1gt_1,102,257,689).  visible(bww1gt_1,689). happensAt(inactive(bww1gt_1),689). orientation(bww1gt_2,60,689). coords(bww1gt_2,237,279,689).  visible(bww1gt_2,689). happensAt(walking(bww1gt_2),689).
orientation(bww1gt_1,173,690). coords(bww1gt_1,102,257,690).  visible(bww1gt_1,690). happensAt(inactive(bww1gt_1),690). orientation(bww1gt_2,60,690). coords(bww1gt_2,237,281,690).  visible(bww1gt_2,690). happensAt(walking(bww1gt_2),690).
orientation(bww1gt_1,173,691). coords(bww1gt_1,102,257,691).  visible(bww1gt_1,691). happensAt(inactive(bww1gt_1),691). orientation(bww1gt_2,60,691). coords(bww1gt_2,237,283,691).  happensAt(disappear(bww1gt_2),691). happensAt(walking(bww1gt_2),691).
#show holdsAt/2.