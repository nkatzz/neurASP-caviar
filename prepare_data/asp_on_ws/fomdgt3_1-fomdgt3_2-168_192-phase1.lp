
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

	orientation(fomdgt3_1,150,168). coords(fomdgt3_1,96,108,168).  visible(fomdgt3_1,168). happensAt(walking(fomdgt3_1),168). orientation(fomdgt3_2,160,168). coords(fomdgt3_2,251,195,168).  visible(fomdgt3_2,168). happensAt(walking(fomdgt3_2),168).
orientation(fomdgt3_1,150,169). coords(fomdgt3_1,92,107,169).  visible(fomdgt3_1,169). happensAt(walking(fomdgt3_1),169). orientation(fomdgt3_2,160,169). coords(fomdgt3_2,251,195,169).  visible(fomdgt3_2,169). happensAt(walking(fomdgt3_2),169).
orientation(fomdgt3_1,150,170). coords(fomdgt3_1,91,107,170).  visible(fomdgt3_1,170). happensAt(walking(fomdgt3_1),170). orientation(fomdgt3_2,160,170). coords(fomdgt3_2,250,193,170).  visible(fomdgt3_2,170). happensAt(walking(fomdgt3_2),170).
orientation(fomdgt3_1,150,171). coords(fomdgt3_1,90,106,171).  visible(fomdgt3_1,171). happensAt(walking(fomdgt3_1),171). orientation(fomdgt3_2,160,171). coords(fomdgt3_2,249,193,171).  visible(fomdgt3_2,171). happensAt(walking(fomdgt3_2),171).
orientation(fomdgt3_1,150,172). coords(fomdgt3_1,90,106,172).  visible(fomdgt3_1,172). happensAt(walking(fomdgt3_1),172). orientation(fomdgt3_2,160,172). coords(fomdgt3_2,249,192,172).  visible(fomdgt3_2,172). happensAt(walking(fomdgt3_2),172).
orientation(fomdgt3_1,150,173). coords(fomdgt3_1,89,105,173).  visible(fomdgt3_1,173). happensAt(walking(fomdgt3_1),173). orientation(fomdgt3_2,160,173). coords(fomdgt3_2,248,189,173).  visible(fomdgt3_2,173). happensAt(walking(fomdgt3_2),173).
orientation(fomdgt3_1,150,174). coords(fomdgt3_1,88,105,174).  visible(fomdgt3_1,174). happensAt(walking(fomdgt3_1),174). orientation(fomdgt3_2,160,174). coords(fomdgt3_2,247,187,174).  visible(fomdgt3_2,174). happensAt(walking(fomdgt3_2),174).
orientation(fomdgt3_1,150,175). coords(fomdgt3_1,89,105,175).  visible(fomdgt3_1,175). happensAt(walking(fomdgt3_1),175). orientation(fomdgt3_2,160,175). coords(fomdgt3_2,245,184,175).  visible(fomdgt3_2,175). happensAt(walking(fomdgt3_2),175).
orientation(fomdgt3_1,150,176). coords(fomdgt3_1,89,105,176).  visible(fomdgt3_1,176). happensAt(walking(fomdgt3_1),176). orientation(fomdgt3_2,160,176). coords(fomdgt3_2,245,184,176).  visible(fomdgt3_2,176). happensAt(walking(fomdgt3_2),176).
orientation(fomdgt3_1,150,177). coords(fomdgt3_1,89,105,177).  visible(fomdgt3_1,177). happensAt(walking(fomdgt3_1),177). orientation(fomdgt3_2,160,177). coords(fomdgt3_2,243,179,177).  visible(fomdgt3_2,177). happensAt(walking(fomdgt3_2),177).
orientation(fomdgt3_1,150,178). coords(fomdgt3_1,88,105,178).  visible(fomdgt3_1,178). happensAt(walking(fomdgt3_1),178). orientation(fomdgt3_2,160,178). coords(fomdgt3_2,242,177,178).  visible(fomdgt3_2,178). happensAt(walking(fomdgt3_2),178).
orientation(fomdgt3_1,150,179). coords(fomdgt3_1,89,105,179).  visible(fomdgt3_1,179). happensAt(walking(fomdgt3_1),179). orientation(fomdgt3_2,160,179). coords(fomdgt3_2,242,176,179).  visible(fomdgt3_2,179). happensAt(walking(fomdgt3_2),179).
orientation(fomdgt3_1,150,180). coords(fomdgt3_1,87,105,180).  visible(fomdgt3_1,180). happensAt(walking(fomdgt3_1),180). orientation(fomdgt3_2,160,180). coords(fomdgt3_2,242,176,180).  visible(fomdgt3_2,180). happensAt(walking(fomdgt3_2),180).
orientation(fomdgt3_1,150,181). coords(fomdgt3_1,86,103,181).  visible(fomdgt3_1,181). happensAt(walking(fomdgt3_1),181). orientation(fomdgt3_2,160,181). coords(fomdgt3_2,242,176,181).  visible(fomdgt3_2,181). happensAt(walking(fomdgt3_2),181).
orientation(fomdgt3_1,150,182). coords(fomdgt3_1,86,103,182).  visible(fomdgt3_1,182). happensAt(walking(fomdgt3_1),182). orientation(fomdgt3_2,160,182). coords(fomdgt3_2,239,172,182).  visible(fomdgt3_2,182). happensAt(walking(fomdgt3_2),182).
orientation(fomdgt3_1,150,183). coords(fomdgt3_1,86,103,183).  visible(fomdgt3_1,183). happensAt(walking(fomdgt3_1),183). orientation(fomdgt3_2,160,183). coords(fomdgt3_2,239,171,183).  visible(fomdgt3_2,183). happensAt(walking(fomdgt3_2),183).
orientation(fomdgt3_1,150,184). coords(fomdgt3_1,83,102,184).  visible(fomdgt3_1,184). happensAt(walking(fomdgt3_1),184). orientation(fomdgt3_2,160,184). coords(fomdgt3_2,239,170,184).  visible(fomdgt3_2,184). happensAt(walking(fomdgt3_2),184).
orientation(fomdgt3_1,150,185). coords(fomdgt3_1,83,102,185).  visible(fomdgt3_1,185). happensAt(walking(fomdgt3_1),185). orientation(fomdgt3_2,130,185). coords(fomdgt3_2,239,168,185).  visible(fomdgt3_2,185). happensAt(walking(fomdgt3_2),185).
orientation(fomdgt3_1,150,186). coords(fomdgt3_1,82,102,186).  visible(fomdgt3_1,186). happensAt(walking(fomdgt3_1),186). orientation(fomdgt3_2,130,186). coords(fomdgt3_2,239,168,186).  visible(fomdgt3_2,186). happensAt(walking(fomdgt3_2),186).
orientation(fomdgt3_1,150,187). coords(fomdgt3_1,82,102,187).  visible(fomdgt3_1,187). happensAt(walking(fomdgt3_1),187). orientation(fomdgt3_2,130,187). coords(fomdgt3_2,239,165,187).  visible(fomdgt3_2,187). happensAt(walking(fomdgt3_2),187).
orientation(fomdgt3_1,150,188). coords(fomdgt3_1,82,102,188).  visible(fomdgt3_1,188). happensAt(walking(fomdgt3_1),188). orientation(fomdgt3_2,130,188). coords(fomdgt3_2,239,164,188).  visible(fomdgt3_2,188). happensAt(walking(fomdgt3_2),188).
orientation(fomdgt3_1,150,189). coords(fomdgt3_1,82,102,189).  visible(fomdgt3_1,189). happensAt(walking(fomdgt3_1),189). orientation(fomdgt3_2,130,189). coords(fomdgt3_2,239,163,189).  visible(fomdgt3_2,189). happensAt(walking(fomdgt3_2),189).
orientation(fomdgt3_1,150,190). coords(fomdgt3_1,82,102,190).  visible(fomdgt3_1,190). happensAt(walking(fomdgt3_1),190). orientation(fomdgt3_2,130,190). coords(fomdgt3_2,237,158,190).  visible(fomdgt3_2,190). happensAt(walking(fomdgt3_2),190).
orientation(fomdgt3_1,150,191). coords(fomdgt3_1,82,102,191).  visible(fomdgt3_1,191). happensAt(walking(fomdgt3_1),191). orientation(fomdgt3_2,130,191). coords(fomdgt3_2,237,156,191).  visible(fomdgt3_2,191). happensAt(walking(fomdgt3_2),191).
orientation(fomdgt3_1,150,192). coords(fomdgt3_1,82,102,192).  visible(fomdgt3_1,192). happensAt(walking(fomdgt3_1),192). orientation(fomdgt3_2,130,192). coords(fomdgt3_2,237,153,192).  visible(fomdgt3_2,192). happensAt(walking(fomdgt3_2),192).
#show holdsAt/2.