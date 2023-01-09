
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

	orientation(wk1gt_0,28,155). coords(wk1gt_0,260,223,155).  visible(wk1gt_0,155). happensAt(walking(wk1gt_0),155). orientation(wk1gt_4,157,155). coords(wk1gt_4,47,183,155).  visible(wk1gt_4,155). happensAt(walking(wk1gt_4),155).
orientation(wk1gt_0,25,156). coords(wk1gt_0,262,227,156).  visible(wk1gt_0,156). happensAt(walking(wk1gt_0),156). orientation(wk1gt_4,164,156). coords(wk1gt_4,47,183,156).  visible(wk1gt_4,156). happensAt(walking(wk1gt_4),156).
orientation(wk1gt_0,22,157). coords(wk1gt_0,264,234,157).  visible(wk1gt_0,157). happensAt(walking(wk1gt_0),157). orientation(wk1gt_4,164,157). coords(wk1gt_4,47,183,157).  visible(wk1gt_4,157). happensAt(walking(wk1gt_4),157).
orientation(wk1gt_0,12,158). coords(wk1gt_0,266,240,158).  visible(wk1gt_0,158). happensAt(walking(wk1gt_0),158). orientation(wk1gt_4,160,158). coords(wk1gt_4,47,183,158).  visible(wk1gt_4,158). happensAt(walking(wk1gt_4),158).
orientation(wk1gt_0,23,159). coords(wk1gt_0,268,244,159).  visible(wk1gt_0,159). happensAt(walking(wk1gt_0),159). orientation(wk1gt_4,160,159). coords(wk1gt_4,47,183,159).  visible(wk1gt_4,159). happensAt(walking(wk1gt_4),159).
orientation(wk1gt_0,31,160). coords(wk1gt_0,270,246,160).  visible(wk1gt_0,160). happensAt(walking(wk1gt_0),160). orientation(wk1gt_4,163,160). coords(wk1gt_4,47,183,160).  visible(wk1gt_4,160). happensAt(walking(wk1gt_4),160).
orientation(wk1gt_0,34,161). coords(wk1gt_0,271,248,161).  visible(wk1gt_0,161). happensAt(walking(wk1gt_0),161). orientation(wk1gt_4,165,161). coords(wk1gt_4,47,183,161).  visible(wk1gt_4,161). happensAt(walking(wk1gt_4),161).
orientation(wk1gt_0,40,162). coords(wk1gt_0,272,250,162).  visible(wk1gt_0,162). happensAt(walking(wk1gt_0),162). orientation(wk1gt_4,161,162). coords(wk1gt_4,47,183,162).  visible(wk1gt_4,162). happensAt(walking(wk1gt_4),162).
orientation(wk1gt_0,43,163). coords(wk1gt_0,274,252,163).  visible(wk1gt_0,163). happensAt(walking(wk1gt_0),163). orientation(wk1gt_4,162,163). coords(wk1gt_4,47,182,163).  visible(wk1gt_4,163). happensAt(walking(wk1gt_4),163).
orientation(wk1gt_0,42,164). coords(wk1gt_0,275,256,164).  visible(wk1gt_0,164). happensAt(walking(wk1gt_0),164). orientation(wk1gt_4,161,164). coords(wk1gt_4,47,182,164).  visible(wk1gt_4,164). happensAt(walking(wk1gt_4),164).
orientation(wk1gt_0,42,165). coords(wk1gt_0,275,260,165).  visible(wk1gt_0,165). happensAt(walking(wk1gt_0),165). orientation(wk1gt_4,163,165). coords(wk1gt_4,47,182,165).  visible(wk1gt_4,165). happensAt(walking(wk1gt_4),165).
orientation(wk1gt_0,52,166). coords(wk1gt_0,276,264,166).  visible(wk1gt_0,166). happensAt(walking(wk1gt_0),166). orientation(wk1gt_4,163,166). coords(wk1gt_4,47,182,166).  visible(wk1gt_4,166). happensAt(walking(wk1gt_4),166).
orientation(wk1gt_0,52,167). coords(wk1gt_0,277,268,167).  visible(wk1gt_0,167). happensAt(walking(wk1gt_0),167). orientation(wk1gt_4,165,167). coords(wk1gt_4,47,182,167).  visible(wk1gt_4,167). happensAt(walking(wk1gt_4),167).
orientation(wk1gt_0,52,168). coords(wk1gt_0,277,271,168).  visible(wk1gt_0,168). happensAt(walking(wk1gt_0),168). orientation(wk1gt_4,159,168). coords(wk1gt_4,47,182,168).  visible(wk1gt_4,168). happensAt(walking(wk1gt_4),168).
orientation(wk1gt_0,35,169). coords(wk1gt_0,277,275,169).  visible(wk1gt_0,169). happensAt(walking(wk1gt_0),169). orientation(wk1gt_4,161,169). coords(wk1gt_4,47,181,169).  visible(wk1gt_4,169). happensAt(walking(wk1gt_4),169).
orientation(wk1gt_0,35,170). coords(wk1gt_0,278,275,170).  visible(wk1gt_0,170). happensAt(walking(wk1gt_0),170). orientation(wk1gt_4,161,170). coords(wk1gt_4,47,181,170).  visible(wk1gt_4,170). happensAt(active(wk1gt_4),170).
orientation(wk1gt_0,35,171). coords(wk1gt_0,278,275,171).  visible(wk1gt_0,171). happensAt(walking(wk1gt_0),171). orientation(wk1gt_4,158,171). coords(wk1gt_4,47,181,171).  visible(wk1gt_4,171). happensAt(active(wk1gt_4),171).
orientation(wk1gt_0,35,172). coords(wk1gt_0,278,275,172).  visible(wk1gt_0,172). happensAt(walking(wk1gt_0),172). orientation(wk1gt_4,165,172). coords(wk1gt_4,47,181,172).  visible(wk1gt_4,172). happensAt(active(wk1gt_4),172).
orientation(wk1gt_0,35,173). coords(wk1gt_0,278,275,173).  visible(wk1gt_0,173). happensAt(walking(wk1gt_0),173). orientation(wk1gt_4,165,173). coords(wk1gt_4,47,181,173).  visible(wk1gt_4,173). happensAt(active(wk1gt_4),173).
orientation(wk1gt_0,35,174). coords(wk1gt_0,278,276,174).  visible(wk1gt_0,174). happensAt(walking(wk1gt_0),174). orientation(wk1gt_4,163,174). coords(wk1gt_4,47,181,174).  visible(wk1gt_4,174). happensAt(active(wk1gt_4),174).
orientation(wk1gt_0,19,175). coords(wk1gt_0,278,278,175).  visible(wk1gt_0,175). happensAt(walking(wk1gt_0),175). orientation(wk1gt_4,162,175). coords(wk1gt_4,47,181,175).  visible(wk1gt_4,175). happensAt(active(wk1gt_4),175).
orientation(wk1gt_0,19,176). coords(wk1gt_0,283,281,176).  visible(wk1gt_0,176). happensAt(walking(wk1gt_0),176). orientation(wk1gt_4,161,176). coords(wk1gt_4,47,181,176).  visible(wk1gt_4,176). happensAt(active(wk1gt_4),176).
orientation(wk1gt_0,19,177). coords(wk1gt_0,285,283,177).  visible(wk1gt_0,177). happensAt(walking(wk1gt_0),177). orientation(wk1gt_4,161,177). coords(wk1gt_4,47,181,177).  visible(wk1gt_4,177). happensAt(active(wk1gt_4),177).
orientation(wk1gt_0,19,178). coords(wk1gt_0,286,285,178).  visible(wk1gt_0,178). happensAt(walking(wk1gt_0),178). orientation(wk1gt_4,163,178). coords(wk1gt_4,47,181,178).  visible(wk1gt_4,178). happensAt(active(wk1gt_4),178).
orientation(wk1gt_0,19,179). coords(wk1gt_0,285,285,179).  happensAt(disappear(wk1gt_0),179). happensAt(walking(wk1gt_0),179). orientation(wk1gt_4,164,179). coords(wk1gt_4,47,181,179).  visible(wk1gt_4,179). happensAt(active(wk1gt_4),179).
#show holdsAt/2.