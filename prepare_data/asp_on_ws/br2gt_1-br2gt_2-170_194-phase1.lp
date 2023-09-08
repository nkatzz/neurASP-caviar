
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

	orientation(br2gt_1,145,170). coords(br2gt_1,220,166,170).  visible(br2gt_1,170). happensAt(walking(br2gt_1),170). orientation(br2gt_2,144,170). coords(br2gt_2,77,59,170).  happensAt(appear(br2gt_2),170). happensAt(walking(br2gt_2),170).
orientation(br2gt_1,145,171). coords(br2gt_1,221,169,171).  visible(br2gt_1,171). happensAt(walking(br2gt_1),171). orientation(br2gt_2,144,171). coords(br2gt_2,77,59,171).  visible(br2gt_2,171). happensAt(walking(br2gt_2),171).
orientation(br2gt_1,145,172). coords(br2gt_1,222,173,172).  visible(br2gt_1,172). happensAt(walking(br2gt_1),172). orientation(br2gt_2,144,172). coords(br2gt_2,77,60,172).  visible(br2gt_2,172). happensAt(walking(br2gt_2),172).
orientation(br2gt_1,145,173). coords(br2gt_1,224,177,173).  visible(br2gt_1,173). happensAt(walking(br2gt_1),173). orientation(br2gt_2,144,173). coords(br2gt_2,77,60,173).  visible(br2gt_2,173). happensAt(walking(br2gt_2),173).
orientation(br2gt_1,145,174). coords(br2gt_1,228,182,174).  visible(br2gt_1,174). happensAt(walking(br2gt_1),174). orientation(br2gt_2,144,174). coords(br2gt_2,77,60,174).  visible(br2gt_2,174). happensAt(walking(br2gt_2),174).
orientation(br2gt_1,145,175). coords(br2gt_1,230,185,175).  visible(br2gt_1,175). happensAt(walking(br2gt_1),175). orientation(br2gt_2,144,175). coords(br2gt_2,77,60,175).  visible(br2gt_2,175). happensAt(walking(br2gt_2),175).
orientation(br2gt_1,145,176). coords(br2gt_1,232,188,176).  visible(br2gt_1,176). happensAt(walking(br2gt_1),176). orientation(br2gt_2,144,176). coords(br2gt_2,77,60,176).  visible(br2gt_2,176). happensAt(walking(br2gt_2),176).
orientation(br2gt_1,142,177). coords(br2gt_1,234,190,177).  visible(br2gt_1,177). happensAt(walking(br2gt_1),177). orientation(br2gt_2,144,177). coords(br2gt_2,76,60,177).  visible(br2gt_2,177). happensAt(walking(br2gt_2),177).
orientation(br2gt_1,144,178). coords(br2gt_1,236,191,178).  visible(br2gt_1,178). happensAt(walking(br2gt_1),178). orientation(br2gt_2,144,178). coords(br2gt_2,77,61,178).  visible(br2gt_2,178). happensAt(walking(br2gt_2),178).
orientation(br2gt_1,146,179). coords(br2gt_1,237,191,179).  visible(br2gt_1,179). happensAt(walking(br2gt_1),179). orientation(br2gt_2,144,179). coords(br2gt_2,76,61,179).  visible(br2gt_2,179). happensAt(walking(br2gt_2),179).
orientation(br2gt_1,149,180). coords(br2gt_1,238,194,180).  visible(br2gt_1,180). happensAt(walking(br2gt_1),180). orientation(br2gt_2,144,180). coords(br2gt_2,76,61,180).  visible(br2gt_2,180). happensAt(walking(br2gt_2),180).
orientation(br2gt_1,149,181). coords(br2gt_1,238,195,181).  visible(br2gt_1,181). happensAt(walking(br2gt_1),181). orientation(br2gt_2,144,181). coords(br2gt_2,76,61,181).  visible(br2gt_2,181). happensAt(walking(br2gt_2),181).
orientation(br2gt_1,149,182). coords(br2gt_1,240,198,182).  visible(br2gt_1,182). happensAt(walking(br2gt_1),182). orientation(br2gt_2,144,182). coords(br2gt_2,76,62,182).  visible(br2gt_2,182). happensAt(walking(br2gt_2),182).
orientation(br2gt_1,149,183). coords(br2gt_1,240,199,183).  visible(br2gt_1,183). happensAt(walking(br2gt_1),183). orientation(br2gt_2,144,183). coords(br2gt_2,76,63,183).  visible(br2gt_2,183). happensAt(walking(br2gt_2),183).
orientation(br2gt_1,160,184). coords(br2gt_1,244,201,184).  visible(br2gt_1,184). happensAt(walking(br2gt_1),184). orientation(br2gt_2,144,184). coords(br2gt_2,76,64,184).  visible(br2gt_2,184). happensAt(walking(br2gt_2),184).
orientation(br2gt_1,165,185). coords(br2gt_1,247,204,185).  visible(br2gt_1,185). happensAt(walking(br2gt_1),185). orientation(br2gt_2,144,185). coords(br2gt_2,76,64,185).  visible(br2gt_2,185). happensAt(walking(br2gt_2),185).
orientation(br2gt_1,171,186). coords(br2gt_1,251,206,186).  visible(br2gt_1,186). happensAt(walking(br2gt_1),186). orientation(br2gt_2,144,186). coords(br2gt_2,76,66,186).  visible(br2gt_2,186). happensAt(walking(br2gt_2),186).
orientation(br2gt_1,176,187). coords(br2gt_1,254,208,187).  visible(br2gt_1,187). happensAt(walking(br2gt_1),187). orientation(br2gt_2,144,187). coords(br2gt_2,76,67,187).  visible(br2gt_2,187). happensAt(walking(br2gt_2),187).
orientation(br2gt_1,9,188). coords(br2gt_1,257,211,188).  visible(br2gt_1,188). happensAt(walking(br2gt_1),188). orientation(br2gt_2,144,188). coords(br2gt_2,76,67,188).  visible(br2gt_2,188). happensAt(walking(br2gt_2),188).
orientation(br2gt_1,9,189). coords(br2gt_1,259,213,189).  visible(br2gt_1,189). happensAt(walking(br2gt_1),189). orientation(br2gt_2,144,189). coords(br2gt_2,76,67,189).  visible(br2gt_2,189). happensAt(walking(br2gt_2),189).
orientation(br2gt_1,9,190). coords(br2gt_1,262,216,190).  visible(br2gt_1,190). happensAt(walking(br2gt_1),190). orientation(br2gt_2,144,190). coords(br2gt_2,76,67,190).  visible(br2gt_2,190). happensAt(walking(br2gt_2),190).
orientation(br2gt_1,18,191). coords(br2gt_1,263,218,191).  visible(br2gt_1,191). happensAt(walking(br2gt_1),191). orientation(br2gt_2,144,191). coords(br2gt_2,76,67,191).  visible(br2gt_2,191). happensAt(walking(br2gt_2),191).
orientation(br2gt_1,18,192). coords(br2gt_1,264,221,192).  visible(br2gt_1,192). happensAt(walking(br2gt_1),192). orientation(br2gt_2,144,192). coords(br2gt_2,76,67,192).  visible(br2gt_2,192). happensAt(walking(br2gt_2),192).
orientation(br2gt_1,18,193). coords(br2gt_1,264,223,193).  visible(br2gt_1,193). happensAt(walking(br2gt_1),193). orientation(br2gt_2,144,193). coords(br2gt_2,76,67,193).  visible(br2gt_2,193). happensAt(walking(br2gt_2),193).
orientation(br2gt_1,18,194). coords(br2gt_1,265,225,194).  visible(br2gt_1,194). happensAt(walking(br2gt_1),194). orientation(br2gt_2,144,194). coords(br2gt_2,75,67,194).  visible(br2gt_2,194). happensAt(walking(br2gt_2),194).
#show holdsAt/2.