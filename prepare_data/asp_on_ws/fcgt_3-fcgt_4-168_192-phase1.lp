
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

	orientation(fcgt_3,159,168). coords(fcgt_3,103,163,168).  visible(fcgt_3,168). happensAt(inactive(fcgt_3),168). orientation(fcgt_4,163,168). coords(fcgt_4,99,209,168).  visible(fcgt_4,168). happensAt(walking(fcgt_4),168).
orientation(fcgt_3,159,169). coords(fcgt_3,103,163,169).  visible(fcgt_3,169). happensAt(inactive(fcgt_3),169). orientation(fcgt_4,163,169). coords(fcgt_4,98,208,169).  visible(fcgt_4,169). happensAt(walking(fcgt_4),169).
orientation(fcgt_3,159,170). coords(fcgt_3,103,163,170).  visible(fcgt_3,170). happensAt(inactive(fcgt_3),170). orientation(fcgt_4,165,170). coords(fcgt_4,98,208,170).  visible(fcgt_4,170). happensAt(walking(fcgt_4),170).
orientation(fcgt_3,159,171). coords(fcgt_3,103,163,171).  visible(fcgt_3,171). happensAt(inactive(fcgt_3),171). orientation(fcgt_4,162,171). coords(fcgt_4,98,208,171).  visible(fcgt_4,171). happensAt(walking(fcgt_4),171).
orientation(fcgt_3,159,172). coords(fcgt_3,103,163,172).  visible(fcgt_3,172). happensAt(inactive(fcgt_3),172). orientation(fcgt_4,163,172). coords(fcgt_4,97,208,172).  visible(fcgt_4,172). happensAt(walking(fcgt_4),172).
orientation(fcgt_3,159,173). coords(fcgt_3,103,163,173).  visible(fcgt_3,173). happensAt(inactive(fcgt_3),173). orientation(fcgt_4,163,173). coords(fcgt_4,96,208,173).  visible(fcgt_4,173). happensAt(walking(fcgt_4),173).
orientation(fcgt_3,159,174). coords(fcgt_3,103,163,174).  visible(fcgt_3,174). happensAt(inactive(fcgt_3),174). orientation(fcgt_4,163,174). coords(fcgt_4,96,207,174).  visible(fcgt_4,174). happensAt(walking(fcgt_4),174).
orientation(fcgt_3,159,175). coords(fcgt_3,103,163,175).  visible(fcgt_3,175). happensAt(inactive(fcgt_3),175). orientation(fcgt_4,163,175). coords(fcgt_4,95,206,175).  visible(fcgt_4,175). happensAt(walking(fcgt_4),175).
orientation(fcgt_3,159,176). coords(fcgt_3,103,163,176).  visible(fcgt_3,176). happensAt(inactive(fcgt_3),176). orientation(fcgt_4,163,176). coords(fcgt_4,94,206,176).  visible(fcgt_4,176). happensAt(walking(fcgt_4),176).
orientation(fcgt_3,159,177). coords(fcgt_3,103,163,177).  visible(fcgt_3,177). happensAt(inactive(fcgt_3),177). orientation(fcgt_4,163,177). coords(fcgt_4,94,206,177).  visible(fcgt_4,177). happensAt(walking(fcgt_4),177).
orientation(fcgt_3,159,178). coords(fcgt_3,103,163,178).  visible(fcgt_3,178). happensAt(inactive(fcgt_3),178). orientation(fcgt_4,163,178). coords(fcgt_4,93,205,178).  visible(fcgt_4,178). happensAt(walking(fcgt_4),178).
orientation(fcgt_3,159,179). coords(fcgt_3,103,163,179).  visible(fcgt_3,179). happensAt(inactive(fcgt_3),179). orientation(fcgt_4,163,179). coords(fcgt_4,93,205,179).  visible(fcgt_4,179). happensAt(walking(fcgt_4),179).
orientation(fcgt_3,159,180). coords(fcgt_3,103,163,180).  visible(fcgt_3,180). happensAt(inactive(fcgt_3),180). orientation(fcgt_4,163,180). coords(fcgt_4,93,205,180).  visible(fcgt_4,180). happensAt(walking(fcgt_4),180).
orientation(fcgt_3,159,181). coords(fcgt_3,103,163,181).  visible(fcgt_3,181). happensAt(inactive(fcgt_3),181). orientation(fcgt_4,163,181). coords(fcgt_4,92,204,181).  visible(fcgt_4,181). happensAt(walking(fcgt_4),181).
orientation(fcgt_3,159,182). coords(fcgt_3,103,163,182).  visible(fcgt_3,182). happensAt(inactive(fcgt_3),182). orientation(fcgt_4,169,182). coords(fcgt_4,92,204,182).  visible(fcgt_4,182). happensAt(walking(fcgt_4),182).
orientation(fcgt_3,159,183). coords(fcgt_3,103,163,183).  visible(fcgt_3,183). happensAt(inactive(fcgt_3),183). orientation(fcgt_4,169,183). coords(fcgt_4,92,204,183).  visible(fcgt_4,183). happensAt(walking(fcgt_4),183).
orientation(fcgt_3,159,184). coords(fcgt_3,103,163,184).  visible(fcgt_3,184). happensAt(inactive(fcgt_3),184). orientation(fcgt_4,169,184). coords(fcgt_4,92,204,184).  visible(fcgt_4,184). happensAt(walking(fcgt_4),184).
orientation(fcgt_3,159,185). coords(fcgt_3,103,163,185).  visible(fcgt_3,185). happensAt(inactive(fcgt_3),185). orientation(fcgt_4,174,185). coords(fcgt_4,92,204,185).  visible(fcgt_4,185). happensAt(walking(fcgt_4),185).
orientation(fcgt_3,159,186). coords(fcgt_3,103,163,186).  visible(fcgt_3,186). happensAt(inactive(fcgt_3),186). orientation(fcgt_4,174,186). coords(fcgt_4,92,204,186).  visible(fcgt_4,186). happensAt(walking(fcgt_4),186).
orientation(fcgt_3,159,187). coords(fcgt_3,103,163,187).  visible(fcgt_3,187). happensAt(inactive(fcgt_3),187). orientation(fcgt_4,174,187). coords(fcgt_4,92,204,187).  visible(fcgt_4,187). happensAt(walking(fcgt_4),187).
orientation(fcgt_3,159,188). coords(fcgt_3,103,163,188).  visible(fcgt_3,188). happensAt(inactive(fcgt_3),188). orientation(fcgt_4,174,188). coords(fcgt_4,91,204,188).  visible(fcgt_4,188). happensAt(walking(fcgt_4),188).
orientation(fcgt_3,159,189). coords(fcgt_3,103,163,189).  visible(fcgt_3,189). happensAt(inactive(fcgt_3),189). orientation(fcgt_4,174,189). coords(fcgt_4,91,204,189).  visible(fcgt_4,189). happensAt(walking(fcgt_4),189).
orientation(fcgt_3,159,190). coords(fcgt_3,103,163,190).  visible(fcgt_3,190). happensAt(inactive(fcgt_3),190). orientation(fcgt_4,174,190). coords(fcgt_4,91,204,190).  visible(fcgt_4,190). happensAt(walking(fcgt_4),190).
orientation(fcgt_3,159,191). coords(fcgt_3,103,163,191).  visible(fcgt_3,191). happensAt(inactive(fcgt_3),191). orientation(fcgt_4,174,191). coords(fcgt_4,91,204,191).  visible(fcgt_4,191). happensAt(walking(fcgt_4),191).
orientation(fcgt_3,159,192). coords(fcgt_3,103,163,192).  visible(fcgt_3,192). happensAt(inactive(fcgt_3),192). orientation(fcgt_4,174,192). coords(fcgt_4,90,204,192).  visible(fcgt_4,192). happensAt(walking(fcgt_4),192).
#show holdsAt/2.