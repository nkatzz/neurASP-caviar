
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

	orientation(wk2gt_1,129,154). coords(wk2gt_1,114,53,154).  visible(wk2gt_1,154). happensAt(inactive(wk2gt_1),154). orientation(wk2gt_3,154,154). coords(wk2gt_3,180,156,154).  visible(wk2gt_3,154). happensAt(inactive(wk2gt_3),154).
orientation(wk2gt_1,129,155). coords(wk2gt_1,114,53,155).  visible(wk2gt_1,155). happensAt(inactive(wk2gt_1),155). orientation(wk2gt_3,149,155). coords(wk2gt_3,180,156,155).  visible(wk2gt_3,155). happensAt(inactive(wk2gt_3),155).
orientation(wk2gt_1,129,156). coords(wk2gt_1,114,53,156).  visible(wk2gt_1,156). happensAt(inactive(wk2gt_1),156). orientation(wk2gt_3,145,156). coords(wk2gt_3,182,156,156).  visible(wk2gt_3,156). happensAt(active(wk2gt_3),156).
orientation(wk2gt_1,129,157). coords(wk2gt_1,114,53,157).  visible(wk2gt_1,157). happensAt(inactive(wk2gt_1),157). orientation(wk2gt_3,145,157). coords(wk2gt_3,183,156,157).  visible(wk2gt_3,157). happensAt(active(wk2gt_3),157).
orientation(wk2gt_1,129,158). coords(wk2gt_1,114,53,158).  visible(wk2gt_1,158). happensAt(inactive(wk2gt_1),158). orientation(wk2gt_3,145,158). coords(wk2gt_3,185,157,158).  visible(wk2gt_3,158). happensAt(active(wk2gt_3),158).
orientation(wk2gt_1,129,159). coords(wk2gt_1,115,53,159).  visible(wk2gt_1,159). happensAt(inactive(wk2gt_1),159). orientation(wk2gt_3,145,159). coords(wk2gt_3,185,157,159).  visible(wk2gt_3,159). happensAt(active(wk2gt_3),159).
orientation(wk2gt_1,129,160). coords(wk2gt_1,115,53,160).  visible(wk2gt_1,160). happensAt(inactive(wk2gt_1),160). orientation(wk2gt_3,145,160). coords(wk2gt_3,186,158,160).  visible(wk2gt_3,160). happensAt(active(wk2gt_3),160).
orientation(wk2gt_1,129,161). coords(wk2gt_1,115,53,161).  visible(wk2gt_1,161). happensAt(inactive(wk2gt_1),161). orientation(wk2gt_3,145,161). coords(wk2gt_3,187,159,161).  visible(wk2gt_3,161). happensAt(active(wk2gt_3),161).
orientation(wk2gt_1,129,162). coords(wk2gt_1,115,53,162).  visible(wk2gt_1,162). happensAt(inactive(wk2gt_1),162). orientation(wk2gt_3,145,162). coords(wk2gt_3,190,161,162).  visible(wk2gt_3,162). happensAt(active(wk2gt_3),162).
orientation(wk2gt_1,129,163). coords(wk2gt_1,115,53,163).  visible(wk2gt_1,163). happensAt(inactive(wk2gt_1),163). orientation(wk2gt_3,145,163). coords(wk2gt_3,191,161,163).  visible(wk2gt_3,163). happensAt(active(wk2gt_3),163).
orientation(wk2gt_1,129,164). coords(wk2gt_1,115,53,164).  visible(wk2gt_1,164). happensAt(inactive(wk2gt_1),164). orientation(wk2gt_3,145,164). coords(wk2gt_3,193,163,164).  visible(wk2gt_3,164). happensAt(active(wk2gt_3),164).
orientation(wk2gt_1,129,165). coords(wk2gt_1,115,53,165).  visible(wk2gt_1,165). happensAt(inactive(wk2gt_1),165). orientation(wk2gt_3,145,165). coords(wk2gt_3,195,165,165).  visible(wk2gt_3,165). happensAt(active(wk2gt_3),165).
orientation(wk2gt_1,129,166). coords(wk2gt_1,115,53,166).  visible(wk2gt_1,166). happensAt(inactive(wk2gt_1),166). orientation(wk2gt_3,145,166). coords(wk2gt_3,196,166,166).  visible(wk2gt_3,166). happensAt(active(wk2gt_3),166).
orientation(wk2gt_1,129,167). coords(wk2gt_1,115,53,167).  visible(wk2gt_1,167). happensAt(inactive(wk2gt_1),167). orientation(wk2gt_3,145,167). coords(wk2gt_3,198,167,167).  visible(wk2gt_3,167). happensAt(active(wk2gt_3),167).
orientation(wk2gt_1,129,168). coords(wk2gt_1,115,53,168).  visible(wk2gt_1,168). happensAt(inactive(wk2gt_1),168). orientation(wk2gt_3,145,168). coords(wk2gt_3,200,168,168).  visible(wk2gt_3,168). happensAt(active(wk2gt_3),168).
orientation(wk2gt_1,129,169). coords(wk2gt_1,115,53,169).  visible(wk2gt_1,169). happensAt(inactive(wk2gt_1),169). orientation(wk2gt_3,145,169). coords(wk2gt_3,201,169,169).  visible(wk2gt_3,169). happensAt(active(wk2gt_3),169).
orientation(wk2gt_1,129,170). coords(wk2gt_1,115,53,170).  visible(wk2gt_1,170). happensAt(inactive(wk2gt_1),170). orientation(wk2gt_3,145,170). coords(wk2gt_3,202,170,170).  visible(wk2gt_3,170). happensAt(active(wk2gt_3),170).
orientation(wk2gt_1,129,171). coords(wk2gt_1,115,53,171).  visible(wk2gt_1,171). happensAt(inactive(wk2gt_1),171). orientation(wk2gt_3,145,171). coords(wk2gt_3,203,172,171).  visible(wk2gt_3,171). happensAt(active(wk2gt_3),171).
orientation(wk2gt_1,129,172). coords(wk2gt_1,115,53,172).  visible(wk2gt_1,172). happensAt(inactive(wk2gt_1),172). orientation(wk2gt_3,145,172). coords(wk2gt_3,205,173,172).  visible(wk2gt_3,172). happensAt(active(wk2gt_3),172).
orientation(wk2gt_1,129,173). coords(wk2gt_1,115,53,173).  visible(wk2gt_1,173). happensAt(inactive(wk2gt_1),173). orientation(wk2gt_3,161,173). coords(wk2gt_3,206,175,173).  visible(wk2gt_3,173). happensAt(active(wk2gt_3),173).
orientation(wk2gt_1,129,174). coords(wk2gt_1,115,53,174).  visible(wk2gt_1,174). happensAt(inactive(wk2gt_1),174). orientation(wk2gt_3,164,174). coords(wk2gt_3,206,175,174).  visible(wk2gt_3,174). happensAt(active(wk2gt_3),174).
orientation(wk2gt_1,129,175). coords(wk2gt_1,115,53,175).  visible(wk2gt_1,175). happensAt(inactive(wk2gt_1),175). orientation(wk2gt_3,164,175). coords(wk2gt_3,207,176,175).  visible(wk2gt_3,175). happensAt(walking(wk2gt_3),175).
orientation(wk2gt_1,129,176). coords(wk2gt_1,115,53,176).  visible(wk2gt_1,176). happensAt(inactive(wk2gt_1),176). orientation(wk2gt_3,164,176). coords(wk2gt_3,208,179,176).  visible(wk2gt_3,176). happensAt(walking(wk2gt_3),176).
orientation(wk2gt_1,129,177). coords(wk2gt_1,115,53,177).  visible(wk2gt_1,177). happensAt(inactive(wk2gt_1),177). orientation(wk2gt_3,164,177). coords(wk2gt_3,210,181,177).  visible(wk2gt_3,177). happensAt(walking(wk2gt_3),177).
orientation(wk2gt_1,129,178). coords(wk2gt_1,115,53,178).  visible(wk2gt_1,178). happensAt(inactive(wk2gt_1),178). orientation(wk2gt_3,164,178). coords(wk2gt_3,210,183,178).  visible(wk2gt_3,178). happensAt(walking(wk2gt_3),178).
#show holdsAt/2.