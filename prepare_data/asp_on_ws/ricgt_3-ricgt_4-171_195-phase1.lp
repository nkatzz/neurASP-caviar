
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

	orientation(ricgt_3,3,171). coords(ricgt_3,236,198,171).  visible(ricgt_3,171). happensAt(walking(ricgt_3),171). orientation(ricgt_4,172,171). coords(ricgt_4,25,141,171).  visible(ricgt_4,171). happensAt(walking(ricgt_4),171).
orientation(ricgt_3,8,172). coords(ricgt_3,238,199,172).  visible(ricgt_3,172). happensAt(walking(ricgt_3),172). orientation(ricgt_4,172,172). coords(ricgt_4,25,141,172).  visible(ricgt_4,172). happensAt(walking(ricgt_4),172).
orientation(ricgt_3,10,173). coords(ricgt_3,238,200,173).  visible(ricgt_3,173). happensAt(walking(ricgt_3),173). orientation(ricgt_4,172,173). coords(ricgt_4,25,141,173).  visible(ricgt_4,173). happensAt(walking(ricgt_4),173).
orientation(ricgt_3,10,174). coords(ricgt_3,239,202,174).  visible(ricgt_3,174). happensAt(walking(ricgt_3),174). orientation(ricgt_4,172,174). coords(ricgt_4,24,141,174).  visible(ricgt_4,174). happensAt(walking(ricgt_4),174).
orientation(ricgt_3,10,175). coords(ricgt_3,240,204,175).  visible(ricgt_3,175). happensAt(walking(ricgt_3),175). orientation(ricgt_4,172,175). coords(ricgt_4,24,141,175).  visible(ricgt_4,175). happensAt(walking(ricgt_4),175).
orientation(ricgt_3,7,176). coords(ricgt_3,241,207,176).  visible(ricgt_3,176). happensAt(walking(ricgt_3),176). orientation(ricgt_4,172,176). coords(ricgt_4,24,141,176).  visible(ricgt_4,176). happensAt(walking(ricgt_4),176).
orientation(ricgt_3,6,177). coords(ricgt_3,242,209,177).  visible(ricgt_3,177). happensAt(walking(ricgt_3),177). orientation(ricgt_4,172,177). coords(ricgt_4,24,141,177).  visible(ricgt_4,177). happensAt(walking(ricgt_4),177).
orientation(ricgt_3,2,178). coords(ricgt_3,243,213,178).  visible(ricgt_3,178). happensAt(walking(ricgt_3),178). orientation(ricgt_4,172,178). coords(ricgt_4,24,141,178).  visible(ricgt_4,178). happensAt(walking(ricgt_4),178).
orientation(ricgt_3,7,179). coords(ricgt_3,247,217,179).  visible(ricgt_3,179). happensAt(walking(ricgt_3),179). orientation(ricgt_4,172,179). coords(ricgt_4,24,141,179).  visible(ricgt_4,179). happensAt(walking(ricgt_4),179).
orientation(ricgt_3,3,180). coords(ricgt_3,250,220,180).  visible(ricgt_3,180). happensAt(walking(ricgt_3),180). orientation(ricgt_4,172,180). coords(ricgt_4,24,141,180).  visible(ricgt_4,180). happensAt(walking(ricgt_4),180).
orientation(ricgt_3,9,181). coords(ricgt_3,253,223,181).  visible(ricgt_3,181). happensAt(walking(ricgt_3),181). orientation(ricgt_4,172,181). coords(ricgt_4,24,141,181).  visible(ricgt_4,181). happensAt(walking(ricgt_4),181).
orientation(ricgt_3,6,182). coords(ricgt_3,256,224,182).  visible(ricgt_3,182). happensAt(walking(ricgt_3),182). orientation(ricgt_4,172,182). coords(ricgt_4,24,141,182).  visible(ricgt_4,182). happensAt(walking(ricgt_4),182).
orientation(ricgt_3,9,183). coords(ricgt_3,258,225,183).  visible(ricgt_3,183). happensAt(walking(ricgt_3),183). orientation(ricgt_4,172,183). coords(ricgt_4,24,142,183).  visible(ricgt_4,183). happensAt(walking(ricgt_4),183).
orientation(ricgt_3,12,184). coords(ricgt_3,259,225,184).  visible(ricgt_3,184). happensAt(walking(ricgt_3),184). orientation(ricgt_4,172,184). coords(ricgt_4,24,142,184).  visible(ricgt_4,184). happensAt(walking(ricgt_4),184).
orientation(ricgt_3,23,185). coords(ricgt_3,260,226,185).  visible(ricgt_3,185). happensAt(walking(ricgt_3),185). orientation(ricgt_4,172,185). coords(ricgt_4,24,143,185).  visible(ricgt_4,185). happensAt(walking(ricgt_4),185).
orientation(ricgt_3,31,186). coords(ricgt_3,261,228,186).  visible(ricgt_3,186). happensAt(walking(ricgt_3),186). orientation(ricgt_4,172,186). coords(ricgt_4,24,143,186).  visible(ricgt_4,186). happensAt(walking(ricgt_4),186).
orientation(ricgt_3,37,187). coords(ricgt_3,262,230,187).  visible(ricgt_3,187). happensAt(walking(ricgt_3),187). orientation(ricgt_4,172,187). coords(ricgt_4,24,143,187).  visible(ricgt_4,187). happensAt(walking(ricgt_4),187).
orientation(ricgt_3,40,188). coords(ricgt_3,263,234,188).  visible(ricgt_3,188). happensAt(walking(ricgt_3),188). orientation(ricgt_4,172,188). coords(ricgt_4,23,143,188).  visible(ricgt_4,188). happensAt(walking(ricgt_4),188).
orientation(ricgt_3,42,189). coords(ricgt_3,265,237,189).  visible(ricgt_3,189). happensAt(walking(ricgt_3),189). orientation(ricgt_4,172,189). coords(ricgt_4,23,143,189).  visible(ricgt_4,189). happensAt(walking(ricgt_4),189).
orientation(ricgt_3,38,190). coords(ricgt_3,268,240,190).  visible(ricgt_3,190). happensAt(walking(ricgt_3),190). orientation(ricgt_4,172,190). coords(ricgt_4,23,144,190).  visible(ricgt_4,190). happensAt(walking(ricgt_4),190).
orientation(ricgt_3,34,191). coords(ricgt_3,273,242,191).  visible(ricgt_3,191). happensAt(walking(ricgt_3),191). orientation(ricgt_4,172,191). coords(ricgt_4,23,145,191).  visible(ricgt_4,191). happensAt(walking(ricgt_4),191).
orientation(ricgt_3,28,192). coords(ricgt_3,277,244,192).  visible(ricgt_3,192). happensAt(walking(ricgt_3),192). orientation(ricgt_4,172,192). coords(ricgt_4,23,145,192).  visible(ricgt_4,192). happensAt(walking(ricgt_4),192).
orientation(ricgt_3,34,193). coords(ricgt_3,280,246,193).  visible(ricgt_3,193). happensAt(walking(ricgt_3),193). orientation(ricgt_4,172,193). coords(ricgt_4,23,146,193).  visible(ricgt_4,193). happensAt(walking(ricgt_4),193).
orientation(ricgt_3,32,194). coords(ricgt_3,283,248,194).  visible(ricgt_3,194). happensAt(walking(ricgt_3),194). orientation(ricgt_4,172,194). coords(ricgt_4,23,146,194).  visible(ricgt_4,194). happensAt(walking(ricgt_4),194).
orientation(ricgt_3,37,195). coords(ricgt_3,285,249,195).  visible(ricgt_3,195). happensAt(walking(ricgt_3),195). orientation(ricgt_4,172,195). coords(ricgt_4,23,147,195).  visible(ricgt_4,195). happensAt(walking(ricgt_4),195).
#show holdsAt/2.