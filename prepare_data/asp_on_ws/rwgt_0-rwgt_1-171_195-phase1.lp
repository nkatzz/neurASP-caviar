
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

	orientation(rwgt_0,149,171). coords(rwgt_0,170,117,171).  visible(rwgt_0,171). happensAt(inactive(rwgt_0),171). orientation(rwgt_1,135,171). coords(rwgt_1,138,56,171).  visible(rwgt_1,171). happensAt(walking(rwgt_1),171).
orientation(rwgt_0,149,172). coords(rwgt_0,170,117,172).  visible(rwgt_0,172). happensAt(inactive(rwgt_0),172). orientation(rwgt_1,135,172). coords(rwgt_1,138,55,172).  visible(rwgt_1,172). happensAt(walking(rwgt_1),172).
orientation(rwgt_0,149,173). coords(rwgt_0,170,117,173).  visible(rwgt_0,173). happensAt(inactive(rwgt_0),173). orientation(rwgt_1,135,173). coords(rwgt_1,138,55,173).  visible(rwgt_1,173). happensAt(walking(rwgt_1),173).
orientation(rwgt_0,149,174). coords(rwgt_0,170,117,174).  visible(rwgt_0,174). happensAt(inactive(rwgt_0),174). orientation(rwgt_1,135,174). coords(rwgt_1,139,54,174).  visible(rwgt_1,174). happensAt(walking(rwgt_1),174).
orientation(rwgt_0,149,175). coords(rwgt_0,170,117,175).  visible(rwgt_0,175). happensAt(inactive(rwgt_0),175). orientation(rwgt_1,135,175). coords(rwgt_1,139,54,175).  visible(rwgt_1,175). happensAt(walking(rwgt_1),175).
orientation(rwgt_0,149,176). coords(rwgt_0,170,117,176).  visible(rwgt_0,176). happensAt(inactive(rwgt_0),176). orientation(rwgt_1,135,176). coords(rwgt_1,138,53,176).  visible(rwgt_1,176). happensAt(walking(rwgt_1),176).
orientation(rwgt_0,149,177). coords(rwgt_0,170,117,177).  visible(rwgt_0,177). happensAt(inactive(rwgt_0),177). orientation(rwgt_1,135,177). coords(rwgt_1,138,52,177).  visible(rwgt_1,177). happensAt(walking(rwgt_1),177).
orientation(rwgt_0,149,178). coords(rwgt_0,170,117,178).  visible(rwgt_0,178). happensAt(inactive(rwgt_0),178). orientation(rwgt_1,135,178). coords(rwgt_1,138,50,178).  visible(rwgt_1,178). happensAt(walking(rwgt_1),178).
orientation(rwgt_0,149,179). coords(rwgt_0,170,117,179).  visible(rwgt_0,179). happensAt(inactive(rwgt_0),179). orientation(rwgt_1,135,179). coords(rwgt_1,138,49,179).  visible(rwgt_1,179). happensAt(walking(rwgt_1),179).
orientation(rwgt_0,149,180). coords(rwgt_0,170,117,180).  visible(rwgt_0,180). happensAt(inactive(rwgt_0),180). orientation(rwgt_1,135,180). coords(rwgt_1,138,47,180).  visible(rwgt_1,180). happensAt(walking(rwgt_1),180).
orientation(rwgt_0,149,181). coords(rwgt_0,170,117,181).  visible(rwgt_0,181). happensAt(inactive(rwgt_0),181). orientation(rwgt_1,135,181). coords(rwgt_1,138,47,181).  visible(rwgt_1,181). happensAt(walking(rwgt_1),181).
orientation(rwgt_0,149,182). coords(rwgt_0,170,117,182).  visible(rwgt_0,182). happensAt(inactive(rwgt_0),182). orientation(rwgt_1,135,182). coords(rwgt_1,138,46,182).  visible(rwgt_1,182). happensAt(walking(rwgt_1),182).
orientation(rwgt_0,149,183). coords(rwgt_0,170,117,183).  visible(rwgt_0,183). happensAt(inactive(rwgt_0),183). orientation(rwgt_1,135,183). coords(rwgt_1,138,45,183).  visible(rwgt_1,183). happensAt(walking(rwgt_1),183).
orientation(rwgt_0,149,184). coords(rwgt_0,170,117,184).  visible(rwgt_0,184). happensAt(inactive(rwgt_0),184). orientation(rwgt_1,135,184). coords(rwgt_1,138,45,184).  visible(rwgt_1,184). happensAt(walking(rwgt_1),184).
orientation(rwgt_0,149,185). coords(rwgt_0,170,117,185).  visible(rwgt_0,185). happensAt(inactive(rwgt_0),185). orientation(rwgt_1,135,185). coords(rwgt_1,138,45,185).  visible(rwgt_1,185). happensAt(walking(rwgt_1),185).
orientation(rwgt_0,149,186). coords(rwgt_0,170,117,186).  visible(rwgt_0,186). happensAt(inactive(rwgt_0),186). orientation(rwgt_1,135,186). coords(rwgt_1,138,45,186).  visible(rwgt_1,186). happensAt(walking(rwgt_1),186).
orientation(rwgt_0,149,187). coords(rwgt_0,171,117,187).  visible(rwgt_0,187). happensAt(inactive(rwgt_0),187). orientation(rwgt_1,135,187). coords(rwgt_1,138,44,187).  visible(rwgt_1,187). happensAt(walking(rwgt_1),187).
orientation(rwgt_0,149,188). coords(rwgt_0,171,117,188).  visible(rwgt_0,188). happensAt(active(rwgt_0),188). orientation(rwgt_1,135,188). coords(rwgt_1,138,43,188).  visible(rwgt_1,188). happensAt(walking(rwgt_1),188).
orientation(rwgt_0,149,189). coords(rwgt_0,171,117,189).  visible(rwgt_0,189). happensAt(active(rwgt_0),189). orientation(rwgt_1,135,189). coords(rwgt_1,139,43,189).  visible(rwgt_1,189). happensAt(walking(rwgt_1),189).
orientation(rwgt_0,149,190). coords(rwgt_0,171,118,190).  visible(rwgt_0,190). happensAt(active(rwgt_0),190). orientation(rwgt_1,135,190). coords(rwgt_1,139,43,190).  visible(rwgt_1,190). happensAt(walking(rwgt_1),190).
orientation(rwgt_0,149,191). coords(rwgt_0,173,119,191).  visible(rwgt_0,191). happensAt(active(rwgt_0),191). orientation(rwgt_1,135,191). coords(rwgt_1,141,43,191).  visible(rwgt_1,191). happensAt(walking(rwgt_1),191).
orientation(rwgt_0,149,192). coords(rwgt_0,175,121,192).  visible(rwgt_0,192). happensAt(active(rwgt_0),192). orientation(rwgt_1,135,192). coords(rwgt_1,141,42,192).  visible(rwgt_1,192). happensAt(walking(rwgt_1),192).
orientation(rwgt_0,149,193). coords(rwgt_0,175,122,193).  visible(rwgt_0,193). happensAt(active(rwgt_0),193). orientation(rwgt_1,135,193). coords(rwgt_1,141,42,193).  visible(rwgt_1,193). happensAt(walking(rwgt_1),193).
orientation(rwgt_0,149,194). coords(rwgt_0,176,125,194).  visible(rwgt_0,194). happensAt(active(rwgt_0),194). orientation(rwgt_1,135,194). coords(rwgt_1,141,42,194).  visible(rwgt_1,194). happensAt(walking(rwgt_1),194).
orientation(rwgt_0,149,195). coords(rwgt_0,177,127,195).  visible(rwgt_0,195). happensAt(active(rwgt_0),195). orientation(rwgt_1,135,195). coords(rwgt_1,141,42,195).  visible(rwgt_1,195). happensAt(walking(rwgt_1),195).
#show holdsAt/2.