
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

	orientation(br1gt_1,164,156). coords(br1gt_1,47,156,156).  visible(br1gt_1,156). happensAt(walking(br1gt_1),156). orientation(br1gt_2,171,156). coords(br1gt_2,86,207,156).  visible(br1gt_2,156). happensAt(inactive(br1gt_2),156).
orientation(br1gt_1,164,157). coords(br1gt_1,47,156,157).  visible(br1gt_1,157). happensAt(walking(br1gt_1),157). orientation(br1gt_2,171,157). coords(br1gt_2,86,207,157).  visible(br1gt_2,157). happensAt(inactive(br1gt_2),157).
orientation(br1gt_1,160,158). coords(br1gt_1,46,156,158).  visible(br1gt_1,158). happensAt(walking(br1gt_1),158). orientation(br1gt_2,171,158). coords(br1gt_2,86,207,158).  visible(br1gt_2,158). happensAt(inactive(br1gt_2),158).
orientation(br1gt_1,160,159). coords(br1gt_1,45,156,159).  visible(br1gt_1,159). happensAt(walking(br1gt_1),159). orientation(br1gt_2,171,159). coords(br1gt_2,86,207,159).  visible(br1gt_2,159). happensAt(inactive(br1gt_2),159).
orientation(br1gt_1,160,160). coords(br1gt_1,45,156,160).  visible(br1gt_1,160). happensAt(walking(br1gt_1),160). orientation(br1gt_2,171,160). coords(br1gt_2,86,207,160).  visible(br1gt_2,160). happensAt(inactive(br1gt_2),160).
orientation(br1gt_1,160,161). coords(br1gt_1,43,156,161).  visible(br1gt_1,161). happensAt(walking(br1gt_1),161). orientation(br1gt_2,171,161). coords(br1gt_2,86,207,161).  visible(br1gt_2,161). happensAt(inactive(br1gt_2),161).
orientation(br1gt_1,160,162). coords(br1gt_1,43,156,162).  visible(br1gt_1,162). happensAt(walking(br1gt_1),162). orientation(br1gt_2,171,162). coords(br1gt_2,86,207,162).  visible(br1gt_2,162). happensAt(inactive(br1gt_2),162).
orientation(br1gt_1,160,163). coords(br1gt_1,43,156,163).  visible(br1gt_1,163). happensAt(walking(br1gt_1),163). orientation(br1gt_2,171,163). coords(br1gt_2,86,207,163).  visible(br1gt_2,163). happensAt(inactive(br1gt_2),163).
orientation(br1gt_1,160,164). coords(br1gt_1,43,156,164).  visible(br1gt_1,164). happensAt(walking(br1gt_1),164). orientation(br1gt_2,171,164). coords(br1gt_2,86,207,164).  visible(br1gt_2,164). happensAt(inactive(br1gt_2),164).
orientation(br1gt_1,160,165). coords(br1gt_1,43,156,165).  visible(br1gt_1,165). happensAt(walking(br1gt_1),165). orientation(br1gt_2,171,165). coords(br1gt_2,86,207,165).  visible(br1gt_2,165). happensAt(inactive(br1gt_2),165).
orientation(br1gt_1,160,166). coords(br1gt_1,43,156,166).  visible(br1gt_1,166). happensAt(walking(br1gt_1),166). orientation(br1gt_2,171,166). coords(br1gt_2,86,207,166).  visible(br1gt_2,166). happensAt(inactive(br1gt_2),166).
orientation(br1gt_1,164,167). coords(br1gt_1,43,156,167).  visible(br1gt_1,167). happensAt(walking(br1gt_1),167). orientation(br1gt_2,171,167). coords(br1gt_2,86,207,167).  visible(br1gt_2,167). happensAt(inactive(br1gt_2),167).
orientation(br1gt_1,164,168). coords(br1gt_1,43,156,168).  visible(br1gt_1,168). happensAt(walking(br1gt_1),168). orientation(br1gt_2,171,168). coords(br1gt_2,86,207,168).  visible(br1gt_2,168). happensAt(inactive(br1gt_2),168).
orientation(br1gt_1,164,169). coords(br1gt_1,42,156,169).  visible(br1gt_1,169). happensAt(walking(br1gt_1),169). orientation(br1gt_2,171,169). coords(br1gt_2,86,207,169).  visible(br1gt_2,169). happensAt(inactive(br1gt_2),169).
orientation(br1gt_1,164,170). coords(br1gt_1,41,156,170).  visible(br1gt_1,170). happensAt(walking(br1gt_1),170). orientation(br1gt_2,171,170). coords(br1gt_2,86,207,170).  visible(br1gt_2,170). happensAt(inactive(br1gt_2),170).
orientation(br1gt_1,164,171). coords(br1gt_1,41,156,171).  visible(br1gt_1,171). happensAt(walking(br1gt_1),171). orientation(br1gt_2,171,171). coords(br1gt_2,86,207,171).  visible(br1gt_2,171). happensAt(inactive(br1gt_2),171).
orientation(br1gt_1,164,172). coords(br1gt_1,40,155,172).  visible(br1gt_1,172). happensAt(walking(br1gt_1),172). orientation(br1gt_2,171,172). coords(br1gt_2,86,207,172).  visible(br1gt_2,172). happensAt(inactive(br1gt_2),172).
orientation(br1gt_1,164,173). coords(br1gt_1,40,155,173).  visible(br1gt_1,173). happensAt(walking(br1gt_1),173). orientation(br1gt_2,171,173). coords(br1gt_2,86,207,173).  visible(br1gt_2,173). happensAt(inactive(br1gt_2),173).
orientation(br1gt_1,164,174). coords(br1gt_1,40,155,174).  visible(br1gt_1,174). happensAt(walking(br1gt_1),174). orientation(br1gt_2,171,174). coords(br1gt_2,86,207,174).  visible(br1gt_2,174). happensAt(inactive(br1gt_2),174).
orientation(br1gt_1,164,175). coords(br1gt_1,40,155,175).  visible(br1gt_1,175). happensAt(walking(br1gt_1),175). orientation(br1gt_2,171,175). coords(br1gt_2,86,207,175).  visible(br1gt_2,175). happensAt(inactive(br1gt_2),175).
orientation(br1gt_1,164,176). coords(br1gt_1,39,155,176).  visible(br1gt_1,176). happensAt(walking(br1gt_1),176). orientation(br1gt_2,171,176). coords(br1gt_2,86,207,176).  visible(br1gt_2,176). happensAt(inactive(br1gt_2),176).
orientation(br1gt_1,164,177). coords(br1gt_1,39,155,177).  visible(br1gt_1,177). happensAt(walking(br1gt_1),177). orientation(br1gt_2,171,177). coords(br1gt_2,86,207,177).  visible(br1gt_2,177). happensAt(inactive(br1gt_2),177).
orientation(br1gt_1,164,178). coords(br1gt_1,39,155,178).  visible(br1gt_1,178). happensAt(walking(br1gt_1),178). orientation(br1gt_2,171,178). coords(br1gt_2,86,207,178).  visible(br1gt_2,178). happensAt(inactive(br1gt_2),178).
orientation(br1gt_1,164,179). coords(br1gt_1,39,155,179).  visible(br1gt_1,179). happensAt(walking(br1gt_1),179). orientation(br1gt_2,171,179). coords(br1gt_2,86,207,179).  visible(br1gt_2,179). happensAt(inactive(br1gt_2),179).
orientation(br1gt_1,164,180). coords(br1gt_1,39,155,180).  visible(br1gt_1,180). happensAt(walking(br1gt_1),180). orientation(br1gt_2,171,180). coords(br1gt_2,86,207,180).  visible(br1gt_2,180). happensAt(inactive(br1gt_2),180).
#show holdsAt/2.