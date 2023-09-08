
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

	orientation(mwt1gt_5,150,673). coords(mwt1gt_5,85,83,673).  visible(mwt1gt_5,673). happensAt(walking(mwt1gt_5),673). orientation(mwt1gt_6,150,673). coords(mwt1gt_6,76,90,673).  visible(mwt1gt_6,673). happensAt(walking(mwt1gt_6),673).
orientation(mwt1gt_5,150,674). coords(mwt1gt_5,85,83,674).  visible(mwt1gt_5,674). happensAt(walking(mwt1gt_5),674). orientation(mwt1gt_6,150,674). coords(mwt1gt_6,76,92,674).  visible(mwt1gt_6,674). happensAt(walking(mwt1gt_6),674).
orientation(mwt1gt_5,150,675). coords(mwt1gt_5,85,83,675).  visible(mwt1gt_5,675). happensAt(walking(mwt1gt_5),675). orientation(mwt1gt_6,150,675). coords(mwt1gt_6,76,92,675).  visible(mwt1gt_6,675). happensAt(walking(mwt1gt_6),675).
orientation(mwt1gt_5,150,676). coords(mwt1gt_5,85,83,676).  visible(mwt1gt_5,676). happensAt(walking(mwt1gt_5),676). orientation(mwt1gt_6,150,676). coords(mwt1gt_6,76,92,676).  visible(mwt1gt_6,676). happensAt(walking(mwt1gt_6),676).
orientation(mwt1gt_5,150,677). coords(mwt1gt_5,85,83,677).  visible(mwt1gt_5,677). happensAt(walking(mwt1gt_5),677). orientation(mwt1gt_6,150,677). coords(mwt1gt_6,76,92,677).  visible(mwt1gt_6,677). happensAt(walking(mwt1gt_6),677).
orientation(mwt1gt_5,150,678). coords(mwt1gt_5,85,83,678).  visible(mwt1gt_5,678). happensAt(walking(mwt1gt_5),678). orientation(mwt1gt_6,150,678). coords(mwt1gt_6,76,92,678).  visible(mwt1gt_6,678). happensAt(walking(mwt1gt_6),678).
orientation(mwt1gt_5,150,679). coords(mwt1gt_5,87,83,679).  visible(mwt1gt_5,679). happensAt(walking(mwt1gt_5),679). orientation(mwt1gt_6,150,679). coords(mwt1gt_6,76,93,679).  visible(mwt1gt_6,679). happensAt(walking(mwt1gt_6),679).
orientation(mwt1gt_5,150,680). coords(mwt1gt_5,89,83,680).  visible(mwt1gt_5,680). happensAt(walking(mwt1gt_5),680). orientation(mwt1gt_6,150,680). coords(mwt1gt_6,76,93,680).  visible(mwt1gt_6,680). happensAt(walking(mwt1gt_6),680).
orientation(mwt1gt_5,150,681). coords(mwt1gt_5,89,83,681).  visible(mwt1gt_5,681). happensAt(walking(mwt1gt_5),681). orientation(mwt1gt_6,150,681). coords(mwt1gt_6,76,93,681).  visible(mwt1gt_6,681). happensAt(walking(mwt1gt_6),681).
orientation(mwt1gt_5,150,682). coords(mwt1gt_5,89,83,682).  visible(mwt1gt_5,682). happensAt(walking(mwt1gt_5),682). orientation(mwt1gt_6,150,682). coords(mwt1gt_6,76,93,682).  visible(mwt1gt_6,682). happensAt(walking(mwt1gt_6),682).
orientation(mwt1gt_5,150,683). coords(mwt1gt_5,91,83,683).  visible(mwt1gt_5,683). happensAt(walking(mwt1gt_5),683). orientation(mwt1gt_6,150,683). coords(mwt1gt_6,77,94,683).  visible(mwt1gt_6,683). happensAt(walking(mwt1gt_6),683).
orientation(mwt1gt_5,150,684). coords(mwt1gt_5,91,83,684).  visible(mwt1gt_5,684). happensAt(walking(mwt1gt_5),684). orientation(mwt1gt_6,150,684). coords(mwt1gt_6,77,94,684).  visible(mwt1gt_6,684). happensAt(walking(mwt1gt_6),684).
orientation(mwt1gt_5,150,685). coords(mwt1gt_5,92,83,685).  visible(mwt1gt_5,685). happensAt(walking(mwt1gt_5),685). orientation(mwt1gt_6,150,685). coords(mwt1gt_6,78,95,685).  visible(mwt1gt_6,685). happensAt(walking(mwt1gt_6),685).
orientation(mwt1gt_5,150,686). coords(mwt1gt_5,92,83,686).  visible(mwt1gt_5,686). happensAt(walking(mwt1gt_5),686). orientation(mwt1gt_6,150,686). coords(mwt1gt_6,78,95,686).  visible(mwt1gt_6,686). happensAt(walking(mwt1gt_6),686).
orientation(mwt1gt_5,150,687). coords(mwt1gt_5,92,83,687).  visible(mwt1gt_5,687). happensAt(walking(mwt1gt_5),687). orientation(mwt1gt_6,150,687). coords(mwt1gt_6,78,95,687).  visible(mwt1gt_6,687). happensAt(walking(mwt1gt_6),687).
orientation(mwt1gt_5,150,688). coords(mwt1gt_5,92,83,688).  visible(mwt1gt_5,688). happensAt(walking(mwt1gt_5),688). orientation(mwt1gt_6,150,688). coords(mwt1gt_6,78,95,688).  visible(mwt1gt_6,688). happensAt(walking(mwt1gt_6),688).
orientation(mwt1gt_5,150,689). coords(mwt1gt_5,92,83,689).  visible(mwt1gt_5,689). happensAt(walking(mwt1gt_5),689). orientation(mwt1gt_6,150,689). coords(mwt1gt_6,78,95,689).  visible(mwt1gt_6,689). happensAt(walking(mwt1gt_6),689).
orientation(mwt1gt_5,150,690). coords(mwt1gt_5,95,83,690).  visible(mwt1gt_5,690). happensAt(walking(mwt1gt_5),690). orientation(mwt1gt_6,150,690). coords(mwt1gt_6,79,96,690).  visible(mwt1gt_6,690). happensAt(walking(mwt1gt_6),690).
orientation(mwt1gt_5,150,691). coords(mwt1gt_5,95,83,691).  visible(mwt1gt_5,691). happensAt(walking(mwt1gt_5),691). orientation(mwt1gt_6,150,691). coords(mwt1gt_6,79,96,691).  visible(mwt1gt_6,691). happensAt(walking(mwt1gt_6),691).
orientation(mwt1gt_5,150,692). coords(mwt1gt_5,95,83,692).  visible(mwt1gt_5,692). happensAt(walking(mwt1gt_5),692). orientation(mwt1gt_6,150,692). coords(mwt1gt_6,79,96,692).  visible(mwt1gt_6,692). happensAt(walking(mwt1gt_6),692).
orientation(mwt1gt_5,150,693). coords(mwt1gt_5,95,83,693).  visible(mwt1gt_5,693). happensAt(walking(mwt1gt_5),693). orientation(mwt1gt_6,150,693). coords(mwt1gt_6,79,96,693).  visible(mwt1gt_6,693). happensAt(walking(mwt1gt_6),693).
orientation(mwt1gt_5,150,694). coords(mwt1gt_5,95,83,694).  visible(mwt1gt_5,694). happensAt(walking(mwt1gt_5),694). orientation(mwt1gt_6,150,694). coords(mwt1gt_6,79,96,694).  visible(mwt1gt_6,694). happensAt(walking(mwt1gt_6),694).
orientation(mwt1gt_5,150,695). coords(mwt1gt_5,96,83,695).  visible(mwt1gt_5,695). happensAt(walking(mwt1gt_5),695). orientation(mwt1gt_6,150,695). coords(mwt1gt_6,80,99,695).  visible(mwt1gt_6,695). happensAt(walking(mwt1gt_6),695).
orientation(mwt1gt_5,150,696). coords(mwt1gt_5,96,83,696).  visible(mwt1gt_5,696). happensAt(walking(mwt1gt_5),696). orientation(mwt1gt_6,150,696). coords(mwt1gt_6,80,99,696).  visible(mwt1gt_6,696). happensAt(walking(mwt1gt_6),696).
orientation(mwt1gt_5,150,697). coords(mwt1gt_5,97,83,697).  visible(mwt1gt_5,697). happensAt(walking(mwt1gt_5),697). orientation(mwt1gt_6,150,697). coords(mwt1gt_6,80,99,697).  visible(mwt1gt_6,697). happensAt(walking(mwt1gt_6),697).
#show holdsAt/2.