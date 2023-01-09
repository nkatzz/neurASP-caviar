
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

	orientation(wk1gt_4,160,159). coords(wk1gt_4,47,183,159).  visible(wk1gt_4,159). happensAt(walking(wk1gt_4),159). orientation(wk1gt_5,162,159). coords(wk1gt_5,46,166,159).  visible(wk1gt_5,159). happensAt(active(wk1gt_5),159).
orientation(wk1gt_4,163,160). coords(wk1gt_4,47,183,160).  visible(wk1gt_4,160). happensAt(walking(wk1gt_4),160). orientation(wk1gt_5,162,160). coords(wk1gt_5,46,166,160).  visible(wk1gt_5,160). happensAt(active(wk1gt_5),160).
orientation(wk1gt_4,165,161). coords(wk1gt_4,47,183,161).  visible(wk1gt_4,161). happensAt(walking(wk1gt_4),161). orientation(wk1gt_5,162,161). coords(wk1gt_5,47,166,161).  visible(wk1gt_5,161). happensAt(active(wk1gt_5),161).
orientation(wk1gt_4,161,162). coords(wk1gt_4,47,183,162).  visible(wk1gt_4,162). happensAt(walking(wk1gt_4),162). orientation(wk1gt_5,162,162). coords(wk1gt_5,47,166,162).  visible(wk1gt_5,162). happensAt(active(wk1gt_5),162).
orientation(wk1gt_4,162,163). coords(wk1gt_4,47,182,163).  visible(wk1gt_4,163). happensAt(walking(wk1gt_4),163). orientation(wk1gt_5,162,163). coords(wk1gt_5,47,166,163).  visible(wk1gt_5,163). happensAt(active(wk1gt_5),163).
orientation(wk1gt_4,161,164). coords(wk1gt_4,47,182,164).  visible(wk1gt_4,164). happensAt(walking(wk1gt_4),164). orientation(wk1gt_5,162,164). coords(wk1gt_5,47,166,164).  visible(wk1gt_5,164). happensAt(active(wk1gt_5),164).
orientation(wk1gt_4,163,165). coords(wk1gt_4,47,182,165).  visible(wk1gt_4,165). happensAt(walking(wk1gt_4),165). orientation(wk1gt_5,162,165). coords(wk1gt_5,47,166,165).  visible(wk1gt_5,165). happensAt(active(wk1gt_5),165).
orientation(wk1gt_4,163,166). coords(wk1gt_4,47,182,166).  visible(wk1gt_4,166). happensAt(walking(wk1gt_4),166). orientation(wk1gt_5,162,166). coords(wk1gt_5,47,166,166).  visible(wk1gt_5,166). happensAt(active(wk1gt_5),166).
orientation(wk1gt_4,165,167). coords(wk1gt_4,47,182,167).  visible(wk1gt_4,167). happensAt(walking(wk1gt_4),167). orientation(wk1gt_5,162,167). coords(wk1gt_5,47,166,167).  visible(wk1gt_5,167). happensAt(active(wk1gt_5),167).
orientation(wk1gt_4,159,168). coords(wk1gt_4,47,182,168).  visible(wk1gt_4,168). happensAt(walking(wk1gt_4),168). orientation(wk1gt_5,162,168). coords(wk1gt_5,48,166,168).  visible(wk1gt_5,168). happensAt(active(wk1gt_5),168).
orientation(wk1gt_4,161,169). coords(wk1gt_4,47,181,169).  visible(wk1gt_4,169). happensAt(walking(wk1gt_4),169). orientation(wk1gt_5,162,169). coords(wk1gt_5,48,166,169).  visible(wk1gt_5,169). happensAt(active(wk1gt_5),169).
orientation(wk1gt_4,161,170). coords(wk1gt_4,47,181,170).  visible(wk1gt_4,170). happensAt(active(wk1gt_4),170). orientation(wk1gt_5,162,170). coords(wk1gt_5,48,166,170).  visible(wk1gt_5,170). happensAt(active(wk1gt_5),170).
orientation(wk1gt_4,158,171). coords(wk1gt_4,47,181,171).  visible(wk1gt_4,171). happensAt(active(wk1gt_4),171). orientation(wk1gt_5,162,171). coords(wk1gt_5,48,166,171).  visible(wk1gt_5,171). happensAt(active(wk1gt_5),171).
orientation(wk1gt_4,165,172). coords(wk1gt_4,47,181,172).  visible(wk1gt_4,172). happensAt(active(wk1gt_4),172). orientation(wk1gt_5,162,172). coords(wk1gt_5,48,166,172).  visible(wk1gt_5,172). happensAt(active(wk1gt_5),172).
orientation(wk1gt_4,165,173). coords(wk1gt_4,47,181,173).  visible(wk1gt_4,173). happensAt(active(wk1gt_4),173). orientation(wk1gt_5,162,173). coords(wk1gt_5,48,166,173).  visible(wk1gt_5,173). happensAt(active(wk1gt_5),173).
orientation(wk1gt_4,163,174). coords(wk1gt_4,47,181,174).  visible(wk1gt_4,174). happensAt(active(wk1gt_4),174). orientation(wk1gt_5,162,174). coords(wk1gt_5,48,166,174).  visible(wk1gt_5,174). happensAt(active(wk1gt_5),174).
orientation(wk1gt_4,162,175). coords(wk1gt_4,47,181,175).  visible(wk1gt_4,175). happensAt(active(wk1gt_4),175). orientation(wk1gt_5,162,175). coords(wk1gt_5,49,166,175).  visible(wk1gt_5,175). happensAt(active(wk1gt_5),175).
orientation(wk1gt_4,161,176). coords(wk1gt_4,47,181,176).  visible(wk1gt_4,176). happensAt(active(wk1gt_4),176). orientation(wk1gt_5,162,176). coords(wk1gt_5,49,166,176).  visible(wk1gt_5,176). happensAt(active(wk1gt_5),176).
orientation(wk1gt_4,161,177). coords(wk1gt_4,47,181,177).  visible(wk1gt_4,177). happensAt(active(wk1gt_4),177). orientation(wk1gt_5,162,177). coords(wk1gt_5,49,166,177).  visible(wk1gt_5,177). happensAt(active(wk1gt_5),177).
orientation(wk1gt_4,163,178). coords(wk1gt_4,47,181,178).  visible(wk1gt_4,178). happensAt(active(wk1gt_4),178). orientation(wk1gt_5,162,178). coords(wk1gt_5,49,166,178).  visible(wk1gt_5,178). happensAt(active(wk1gt_5),178).
orientation(wk1gt_4,164,179). coords(wk1gt_4,47,181,179).  visible(wk1gt_4,179). happensAt(active(wk1gt_4),179). orientation(wk1gt_5,162,179). coords(wk1gt_5,49,166,179).  visible(wk1gt_5,179). happensAt(active(wk1gt_5),179).
orientation(wk1gt_4,160,180). coords(wk1gt_4,47,181,180).  visible(wk1gt_4,180). happensAt(inactive(wk1gt_4),180). orientation(wk1gt_5,162,180). coords(wk1gt_5,49,166,180).  visible(wk1gt_5,180). happensAt(active(wk1gt_5),180).
orientation(wk1gt_4,158,181). coords(wk1gt_4,47,181,181).  visible(wk1gt_4,181). happensAt(inactive(wk1gt_4),181). orientation(wk1gt_5,162,181). coords(wk1gt_5,49,166,181).  visible(wk1gt_5,181). happensAt(active(wk1gt_5),181).
orientation(wk1gt_4,163,182). coords(wk1gt_4,47,181,182).  visible(wk1gt_4,182). happensAt(inactive(wk1gt_4),182). orientation(wk1gt_5,162,182). coords(wk1gt_5,49,166,182).  visible(wk1gt_5,182). happensAt(active(wk1gt_5),182).
orientation(wk1gt_4,162,183). coords(wk1gt_4,47,181,183).  visible(wk1gt_4,183). happensAt(inactive(wk1gt_4),183). orientation(wk1gt_5,162,183). coords(wk1gt_5,49,166,183).  visible(wk1gt_5,183). happensAt(active(wk1gt_5),183).
#show holdsAt/2.