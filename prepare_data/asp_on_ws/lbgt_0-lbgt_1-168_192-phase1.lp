
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

	orientation(lbgt_0,168,168). coords(lbgt_0,206,202,168).  visible(lbgt_0,168). happensAt(active(lbgt_0),168). orientation(lbgt_1,152,168). coords(lbgt_1,79,98,168).  visible(lbgt_1,168). happensAt(walking(lbgt_1),168).
orientation(lbgt_0,168,169). coords(lbgt_0,206,202,169).  visible(lbgt_0,169). happensAt(active(lbgt_0),169). orientation(lbgt_1,152,169). coords(lbgt_1,79,98,169).  visible(lbgt_1,169). happensAt(walking(lbgt_1),169).
orientation(lbgt_0,168,170). coords(lbgt_0,206,201,170).  visible(lbgt_0,170). happensAt(active(lbgt_0),170). orientation(lbgt_1,152,170). coords(lbgt_1,79,98,170).  visible(lbgt_1,170). happensAt(walking(lbgt_1),170).
orientation(lbgt_0,0,171). coords(lbgt_0,206,200,171).  visible(lbgt_0,171). happensAt(active(lbgt_0),171). orientation(lbgt_1,152,171). coords(lbgt_1,79,98,171).  visible(lbgt_1,171). happensAt(walking(lbgt_1),171).
orientation(lbgt_0,0,172). coords(lbgt_0,206,197,172).  visible(lbgt_0,172). happensAt(active(lbgt_0),172). orientation(lbgt_1,152,172). coords(lbgt_1,79,98,172).  visible(lbgt_1,172). happensAt(walking(lbgt_1),172).
orientation(lbgt_0,0,173). coords(lbgt_0,207,198,173).  visible(lbgt_0,173). happensAt(active(lbgt_0),173). orientation(lbgt_1,152,173). coords(lbgt_1,79,98,173).  visible(lbgt_1,173). happensAt(walking(lbgt_1),173).
orientation(lbgt_0,0,174). coords(lbgt_0,208,197,174).  visible(lbgt_0,174). happensAt(active(lbgt_0),174). orientation(lbgt_1,152,174). coords(lbgt_1,79,98,174).  visible(lbgt_1,174). happensAt(walking(lbgt_1),174).
orientation(lbgt_0,0,175). coords(lbgt_0,208,197,175).  visible(lbgt_0,175). happensAt(active(lbgt_0),175). orientation(lbgt_1,152,175). coords(lbgt_1,79,99,175).  visible(lbgt_1,175). happensAt(walking(lbgt_1),175).
orientation(lbgt_0,0,176). coords(lbgt_0,211,196,176).  visible(lbgt_0,176). happensAt(active(lbgt_0),176). orientation(lbgt_1,152,176). coords(lbgt_1,79,98,176).  visible(lbgt_1,176). happensAt(walking(lbgt_1),176).
orientation(lbgt_0,0,177). coords(lbgt_0,213,196,177).  visible(lbgt_0,177). happensAt(active(lbgt_0),177). orientation(lbgt_1,152,177). coords(lbgt_1,79,99,177).  visible(lbgt_1,177). happensAt(walking(lbgt_1),177).
orientation(lbgt_0,176,178). coords(lbgt_0,214,195,178).  visible(lbgt_0,178). happensAt(walking(lbgt_0),178). orientation(lbgt_1,152,178). coords(lbgt_1,79,98,178).  visible(lbgt_1,178). happensAt(walking(lbgt_1),178).
orientation(lbgt_0,171,179). coords(lbgt_0,214,195,179).  visible(lbgt_0,179). happensAt(walking(lbgt_0),179). orientation(lbgt_1,152,179). coords(lbgt_1,79,98,179).  visible(lbgt_1,179). happensAt(walking(lbgt_1),179).
orientation(lbgt_0,171,180). coords(lbgt_0,213,194,180).  visible(lbgt_0,180). happensAt(walking(lbgt_0),180). orientation(lbgt_1,152,180). coords(lbgt_1,79,98,180).  visible(lbgt_1,180). happensAt(walking(lbgt_1),180).
orientation(lbgt_0,168,181). coords(lbgt_0,212,193,181).  visible(lbgt_0,181). happensAt(walking(lbgt_0),181). orientation(lbgt_1,152,181). coords(lbgt_1,79,98,181).  visible(lbgt_1,181). happensAt(walking(lbgt_1),181).
orientation(lbgt_0,168,182). coords(lbgt_0,210,194,182).  visible(lbgt_0,182). happensAt(walking(lbgt_0),182). orientation(lbgt_1,152,182). coords(lbgt_1,79,98,182).  visible(lbgt_1,182). happensAt(walking(lbgt_1),182).
orientation(lbgt_0,168,183). coords(lbgt_0,209,192,183).  visible(lbgt_0,183). happensAt(walking(lbgt_0),183). orientation(lbgt_1,152,183). coords(lbgt_1,79,97,183).  visible(lbgt_1,183). happensAt(walking(lbgt_1),183).
orientation(lbgt_0,168,184). coords(lbgt_0,208,191,184).  visible(lbgt_0,184). happensAt(walking(lbgt_0),184). orientation(lbgt_1,152,184). coords(lbgt_1,79,97,184).  visible(lbgt_1,184). happensAt(walking(lbgt_1),184).
orientation(lbgt_0,168,185). coords(lbgt_0,208,189,185).  visible(lbgt_0,185). happensAt(walking(lbgt_0),185). orientation(lbgt_1,152,185). coords(lbgt_1,79,97,185).  visible(lbgt_1,185). happensAt(walking(lbgt_1),185).
orientation(lbgt_0,160,186). coords(lbgt_0,208,188,186).  visible(lbgt_0,186). happensAt(walking(lbgt_0),186). orientation(lbgt_1,152,186). coords(lbgt_1,79,97,186).  visible(lbgt_1,186). happensAt(walking(lbgt_1),186).
orientation(lbgt_0,160,187). coords(lbgt_0,207,186,187).  visible(lbgt_0,187). happensAt(walking(lbgt_0),187). orientation(lbgt_1,152,187). coords(lbgt_1,79,97,187).  visible(lbgt_1,187). happensAt(walking(lbgt_1),187).
orientation(lbgt_0,160,188). coords(lbgt_0,207,185,188).  visible(lbgt_0,188). happensAt(walking(lbgt_0),188). orientation(lbgt_1,152,188). coords(lbgt_1,79,97,188).  visible(lbgt_1,188). happensAt(walking(lbgt_1),188).
orientation(lbgt_0,160,189). coords(lbgt_0,207,185,189).  visible(lbgt_0,189). happensAt(walking(lbgt_0),189). orientation(lbgt_1,152,189). coords(lbgt_1,79,96,189).  visible(lbgt_1,189). happensAt(walking(lbgt_1),189).
orientation(lbgt_0,160,190). coords(lbgt_0,207,184,190).  visible(lbgt_0,190). happensAt(walking(lbgt_0),190). orientation(lbgt_1,152,190). coords(lbgt_1,79,95,190).  visible(lbgt_1,190). happensAt(walking(lbgt_1),190).
orientation(lbgt_0,160,191). coords(lbgt_0,206,183,191).  visible(lbgt_0,191). happensAt(walking(lbgt_0),191). orientation(lbgt_1,152,191). coords(lbgt_1,79,96,191).  visible(lbgt_1,191). happensAt(walking(lbgt_1),191).
orientation(lbgt_0,156,192). coords(lbgt_0,206,182,192).  visible(lbgt_0,192). happensAt(walking(lbgt_0),192). orientation(lbgt_1,152,192). coords(lbgt_1,79,95,192).  visible(lbgt_1,192). happensAt(walking(lbgt_1),192).
#show holdsAt/2.