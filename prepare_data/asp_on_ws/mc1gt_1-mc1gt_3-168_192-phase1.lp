
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

	orientation(mc1gt_1,145,168). coords(mc1gt_1,133,87,168).  visible(mc1gt_1,168). happensAt(walking(mc1gt_1),168). orientation(mc1gt_3,146,168). coords(mc1gt_3,105,81,168).  visible(mc1gt_3,168). happensAt(walking(mc1gt_3),168).
orientation(mc1gt_1,145,169). coords(mc1gt_1,134,87,169).  visible(mc1gt_1,169). happensAt(walking(mc1gt_1),169). orientation(mc1gt_3,146,169). coords(mc1gt_3,105,81,169).  visible(mc1gt_3,169). happensAt(walking(mc1gt_3),169).
orientation(mc1gt_1,145,170). coords(mc1gt_1,136,88,170).  visible(mc1gt_1,170). happensAt(walking(mc1gt_1),170). orientation(mc1gt_3,146,170). coords(mc1gt_3,105,81,170).  visible(mc1gt_3,170). happensAt(walking(mc1gt_3),170).
orientation(mc1gt_1,145,171). coords(mc1gt_1,137,89,171).  visible(mc1gt_1,171). happensAt(walking(mc1gt_1),171). orientation(mc1gt_3,146,171). coords(mc1gt_3,106,81,171).  visible(mc1gt_3,171). happensAt(walking(mc1gt_3),171).
orientation(mc1gt_1,145,172). coords(mc1gt_1,138,90,172).  visible(mc1gt_1,172). happensAt(walking(mc1gt_1),172). orientation(mc1gt_3,146,172). coords(mc1gt_3,107,81,172).  visible(mc1gt_3,172). happensAt(walking(mc1gt_3),172).
orientation(mc1gt_1,145,173). coords(mc1gt_1,138,90,173).  visible(mc1gt_1,173). happensAt(walking(mc1gt_1),173). orientation(mc1gt_3,146,173). coords(mc1gt_3,108,81,173).  visible(mc1gt_3,173). happensAt(walking(mc1gt_3),173).
orientation(mc1gt_1,145,174). coords(mc1gt_1,138,90,174).  visible(mc1gt_1,174). happensAt(walking(mc1gt_1),174). orientation(mc1gt_3,146,174). coords(mc1gt_3,109,81,174).  visible(mc1gt_3,174). happensAt(walking(mc1gt_3),174).
orientation(mc1gt_1,145,175). coords(mc1gt_1,139,90,175).  visible(mc1gt_1,175). happensAt(walking(mc1gt_1),175). orientation(mc1gt_3,146,175). coords(mc1gt_3,110,81,175).  visible(mc1gt_3,175). happensAt(walking(mc1gt_3),175).
orientation(mc1gt_1,145,176). coords(mc1gt_1,139,90,176).  visible(mc1gt_1,176). happensAt(walking(mc1gt_1),176). orientation(mc1gt_3,146,176). coords(mc1gt_3,112,81,176).  visible(mc1gt_3,176). happensAt(walking(mc1gt_3),176).
orientation(mc1gt_1,145,177). coords(mc1gt_1,139,91,177).  visible(mc1gt_1,177). happensAt(walking(mc1gt_1),177). orientation(mc1gt_3,146,177). coords(mc1gt_3,112,81,177).  visible(mc1gt_3,177). happensAt(walking(mc1gt_3),177).
orientation(mc1gt_1,145,178). coords(mc1gt_1,139,91,178).  visible(mc1gt_1,178). happensAt(walking(mc1gt_1),178). orientation(mc1gt_3,146,178). coords(mc1gt_3,112,81,178).  visible(mc1gt_3,178). happensAt(walking(mc1gt_3),178).
orientation(mc1gt_1,145,179). coords(mc1gt_1,140,91,179).  visible(mc1gt_1,179). happensAt(walking(mc1gt_1),179). orientation(mc1gt_3,146,179). coords(mc1gt_3,113,81,179).  visible(mc1gt_3,179). happensAt(walking(mc1gt_3),179).
orientation(mc1gt_1,145,180). coords(mc1gt_1,140,91,180).  visible(mc1gt_1,180). happensAt(walking(mc1gt_1),180). orientation(mc1gt_3,146,180). coords(mc1gt_3,113,81,180).  visible(mc1gt_3,180). happensAt(walking(mc1gt_3),180).
orientation(mc1gt_1,145,181). coords(mc1gt_1,140,92,181).  visible(mc1gt_1,181). happensAt(walking(mc1gt_1),181). orientation(mc1gt_3,146,181). coords(mc1gt_3,113,82,181).  visible(mc1gt_3,181). happensAt(walking(mc1gt_3),181).
orientation(mc1gt_1,145,182). coords(mc1gt_1,142,92,182).  visible(mc1gt_1,182). happensAt(walking(mc1gt_1),182). orientation(mc1gt_3,146,182). coords(mc1gt_3,114,82,182).  visible(mc1gt_3,182). happensAt(walking(mc1gt_3),182).
orientation(mc1gt_1,145,183). coords(mc1gt_1,144,92,183).  visible(mc1gt_1,183). happensAt(walking(mc1gt_1),183). orientation(mc1gt_3,146,183). coords(mc1gt_3,114,82,183).  visible(mc1gt_3,183). happensAt(walking(mc1gt_3),183).
orientation(mc1gt_1,145,184). coords(mc1gt_1,146,92,184).  visible(mc1gt_1,184). happensAt(walking(mc1gt_1),184). orientation(mc1gt_3,146,184). coords(mc1gt_3,115,84,184).  visible(mc1gt_3,184). happensAt(walking(mc1gt_3),184).
orientation(mc1gt_1,145,185). coords(mc1gt_1,147,94,185).  visible(mc1gt_1,185). happensAt(walking(mc1gt_1),185). orientation(mc1gt_3,146,185). coords(mc1gt_3,115,84,185).  visible(mc1gt_3,185). happensAt(walking(mc1gt_3),185).
orientation(mc1gt_1,145,186). coords(mc1gt_1,148,94,186).  visible(mc1gt_1,186). happensAt(walking(mc1gt_1),186). orientation(mc1gt_3,146,186). coords(mc1gt_3,117,84,186).  visible(mc1gt_3,186). happensAt(walking(mc1gt_3),186).
orientation(mc1gt_1,145,187). coords(mc1gt_1,149,95,187).  visible(mc1gt_1,187). happensAt(walking(mc1gt_1),187). orientation(mc1gt_3,146,187). coords(mc1gt_3,119,85,187).  visible(mc1gt_3,187). happensAt(walking(mc1gt_3),187).
orientation(mc1gt_1,143,188). coords(mc1gt_1,150,96,188).  visible(mc1gt_1,188). happensAt(walking(mc1gt_1),188). orientation(mc1gt_3,146,188). coords(mc1gt_3,119,83,188).  visible(mc1gt_3,188). happensAt(walking(mc1gt_3),188).
orientation(mc1gt_1,141,189). coords(mc1gt_1,151,96,189).  visible(mc1gt_1,189). happensAt(walking(mc1gt_1),189). orientation(mc1gt_3,146,189). coords(mc1gt_3,118,84,189).  visible(mc1gt_3,189). happensAt(walking(mc1gt_3),189).
orientation(mc1gt_1,139,190). coords(mc1gt_1,151,97,190).  visible(mc1gt_1,190). happensAt(walking(mc1gt_1),190). orientation(mc1gt_3,146,190). coords(mc1gt_3,117,82,190).  visible(mc1gt_3,190). happensAt(walking(mc1gt_3),190).
orientation(mc1gt_1,139,191). coords(mc1gt_1,152,97,191).  visible(mc1gt_1,191). happensAt(walking(mc1gt_1),191). orientation(mc1gt_3,146,191). coords(mc1gt_3,118,82,191).  visible(mc1gt_3,191). happensAt(walking(mc1gt_3),191).
orientation(mc1gt_1,140,192). coords(mc1gt_1,152,97,192).  visible(mc1gt_1,192). happensAt(walking(mc1gt_1),192). orientation(mc1gt_3,146,192). coords(mc1gt_3,118,82,192).  visible(mc1gt_3,192). happensAt(walking(mc1gt_3),192).
#show holdsAt/2.