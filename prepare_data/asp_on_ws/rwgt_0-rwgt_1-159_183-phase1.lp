
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

	orientation(rwgt_0,149,159). coords(rwgt_0,171,117,159).  visible(rwgt_0,159). happensAt(active(rwgt_0),159). orientation(rwgt_1,135,159). coords(rwgt_1,141,66,159).  visible(rwgt_1,159). happensAt(walking(rwgt_1),159).
orientation(rwgt_0,149,160). coords(rwgt_0,170,117,160).  visible(rwgt_0,160). happensAt(inactive(rwgt_0),160). orientation(rwgt_1,135,160). coords(rwgt_1,141,65,160).  visible(rwgt_1,160). happensAt(walking(rwgt_1),160).
orientation(rwgt_0,149,161). coords(rwgt_0,170,117,161).  visible(rwgt_0,161). happensAt(inactive(rwgt_0),161). orientation(rwgt_1,135,161). coords(rwgt_1,141,64,161).  visible(rwgt_1,161). happensAt(walking(rwgt_1),161).
orientation(rwgt_0,149,162). coords(rwgt_0,170,117,162).  visible(rwgt_0,162). happensAt(inactive(rwgt_0),162). orientation(rwgt_1,135,162). coords(rwgt_1,141,64,162).  visible(rwgt_1,162). happensAt(walking(rwgt_1),162).
orientation(rwgt_0,149,163). coords(rwgt_0,170,117,163).  visible(rwgt_0,163). happensAt(inactive(rwgt_0),163). orientation(rwgt_1,135,163). coords(rwgt_1,141,63,163).  visible(rwgt_1,163). happensAt(walking(rwgt_1),163).
orientation(rwgt_0,149,164). coords(rwgt_0,170,117,164).  visible(rwgt_0,164). happensAt(inactive(rwgt_0),164). orientation(rwgt_1,135,164). coords(rwgt_1,140,62,164).  visible(rwgt_1,164). happensAt(walking(rwgt_1),164).
orientation(rwgt_0,149,165). coords(rwgt_0,170,117,165).  visible(rwgt_0,165). happensAt(inactive(rwgt_0),165). orientation(rwgt_1,135,165). coords(rwgt_1,140,61,165).  visible(rwgt_1,165). happensAt(walking(rwgt_1),165).
orientation(rwgt_0,149,166). coords(rwgt_0,170,117,166).  visible(rwgt_0,166). happensAt(inactive(rwgt_0),166). orientation(rwgt_1,135,166). coords(rwgt_1,140,59,166).  visible(rwgt_1,166). happensAt(walking(rwgt_1),166).
orientation(rwgt_0,149,167). coords(rwgt_0,170,117,167).  visible(rwgt_0,167). happensAt(inactive(rwgt_0),167). orientation(rwgt_1,135,167). coords(rwgt_1,139,58,167).  visible(rwgt_1,167). happensAt(walking(rwgt_1),167).
orientation(rwgt_0,149,168). coords(rwgt_0,170,117,168).  visible(rwgt_0,168). happensAt(inactive(rwgt_0),168). orientation(rwgt_1,135,168). coords(rwgt_1,138,57,168).  visible(rwgt_1,168). happensAt(walking(rwgt_1),168).
orientation(rwgt_0,149,169). coords(rwgt_0,170,117,169).  visible(rwgt_0,169). happensAt(inactive(rwgt_0),169). orientation(rwgt_1,135,169). coords(rwgt_1,138,56,169).  visible(rwgt_1,169). happensAt(walking(rwgt_1),169).
orientation(rwgt_0,149,170). coords(rwgt_0,170,117,170).  visible(rwgt_0,170). happensAt(inactive(rwgt_0),170). orientation(rwgt_1,135,170). coords(rwgt_1,138,56,170).  visible(rwgt_1,170). happensAt(walking(rwgt_1),170).
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
#show holdsAt/2.