
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

	orientation(br1gt_0,160,156). coords(br1gt_0,252,216,156).  visible(br1gt_0,156). happensAt(walking(br1gt_0),156). orientation(br1gt_2,171,156). coords(br1gt_2,86,207,156).  visible(br1gt_2,156). happensAt(inactive(br1gt_2),156).
orientation(br1gt_0,156,157). coords(br1gt_0,254,220,157).  visible(br1gt_0,157). happensAt(walking(br1gt_0),157). orientation(br1gt_2,171,157). coords(br1gt_2,86,207,157).  visible(br1gt_2,157). happensAt(inactive(br1gt_2),157).
orientation(br1gt_0,156,158). coords(br1gt_0,257,224,158).  visible(br1gt_0,158). happensAt(walking(br1gt_0),158). orientation(br1gt_2,171,158). coords(br1gt_2,86,207,158).  visible(br1gt_2,158). happensAt(inactive(br1gt_2),158).
orientation(br1gt_0,156,159). coords(br1gt_0,259,226,159).  visible(br1gt_0,159). happensAt(walking(br1gt_0),159). orientation(br1gt_2,171,159). coords(br1gt_2,86,207,159).  visible(br1gt_2,159). happensAt(inactive(br1gt_2),159).
orientation(br1gt_0,156,160). coords(br1gt_0,262,227,160).  visible(br1gt_0,160). happensAt(walking(br1gt_0),160). orientation(br1gt_2,171,160). coords(br1gt_2,86,207,160).  visible(br1gt_2,160). happensAt(inactive(br1gt_2),160).
orientation(br1gt_0,156,161). coords(br1gt_0,264,227,161).  visible(br1gt_0,161). happensAt(walking(br1gt_0),161). orientation(br1gt_2,171,161). coords(br1gt_2,86,207,161).  visible(br1gt_2,161). happensAt(inactive(br1gt_2),161).
orientation(br1gt_0,52,162). coords(br1gt_0,266,227,162).  visible(br1gt_0,162). happensAt(walking(br1gt_0),162). orientation(br1gt_2,171,162). coords(br1gt_2,86,207,162).  visible(br1gt_2,162). happensAt(inactive(br1gt_2),162).
orientation(br1gt_0,52,163). coords(br1gt_0,266,228,163).  visible(br1gt_0,163). happensAt(walking(br1gt_0),163). orientation(br1gt_2,171,163). coords(br1gt_2,86,207,163).  visible(br1gt_2,163). happensAt(inactive(br1gt_2),163).
orientation(br1gt_0,52,164). coords(br1gt_0,268,230,164).  visible(br1gt_0,164). happensAt(walking(br1gt_0),164). orientation(br1gt_2,171,164). coords(br1gt_2,86,207,164).  visible(br1gt_2,164). happensAt(inactive(br1gt_2),164).
orientation(br1gt_0,52,165). coords(br1gt_0,269,233,165).  visible(br1gt_0,165). happensAt(walking(br1gt_0),165). orientation(br1gt_2,171,165). coords(br1gt_2,86,207,165).  visible(br1gt_2,165). happensAt(inactive(br1gt_2),165).
orientation(br1gt_0,47,166). coords(br1gt_0,269,233,166).  visible(br1gt_0,166). happensAt(walking(br1gt_0),166). orientation(br1gt_2,171,166). coords(br1gt_2,86,207,166).  visible(br1gt_2,166). happensAt(inactive(br1gt_2),166).
orientation(br1gt_0,44,167). coords(br1gt_0,271,240,167).  visible(br1gt_0,167). happensAt(walking(br1gt_0),167). orientation(br1gt_2,171,167). coords(br1gt_2,86,207,167).  visible(br1gt_2,167). happensAt(inactive(br1gt_2),167).
orientation(br1gt_0,44,168). coords(br1gt_0,273,243,168).  visible(br1gt_0,168). happensAt(walking(br1gt_0),168). orientation(br1gt_2,171,168). coords(br1gt_2,86,207,168).  visible(br1gt_2,168). happensAt(inactive(br1gt_2),168).
orientation(br1gt_0,33,169). coords(br1gt_0,277,245,169).  visible(br1gt_0,169). happensAt(walking(br1gt_0),169). orientation(br1gt_2,171,169). coords(br1gt_2,86,207,169).  visible(br1gt_2,169). happensAt(inactive(br1gt_2),169).
orientation(br1gt_0,33,170). coords(br1gt_0,281,246,170).  visible(br1gt_0,170). happensAt(walking(br1gt_0),170). orientation(br1gt_2,171,170). coords(br1gt_2,86,207,170).  visible(br1gt_2,170). happensAt(inactive(br1gt_2),170).
orientation(br1gt_0,33,171). coords(br1gt_0,284,248,171).  visible(br1gt_0,171). happensAt(walking(br1gt_0),171). orientation(br1gt_2,171,171). coords(br1gt_2,86,207,171).  visible(br1gt_2,171). happensAt(inactive(br1gt_2),171).
orientation(br1gt_0,33,172). coords(br1gt_0,286,249,172).  visible(br1gt_0,172). happensAt(walking(br1gt_0),172). orientation(br1gt_2,171,172). coords(br1gt_2,86,207,172).  visible(br1gt_2,172). happensAt(inactive(br1gt_2),172).
orientation(br1gt_0,33,173). coords(br1gt_0,289,251,173).  visible(br1gt_0,173). happensAt(walking(br1gt_0),173). orientation(br1gt_2,171,173). coords(br1gt_2,86,207,173).  visible(br1gt_2,173). happensAt(inactive(br1gt_2),173).
orientation(br1gt_0,33,174). coords(br1gt_0,291,253,174).  visible(br1gt_0,174). happensAt(walking(br1gt_0),174). orientation(br1gt_2,171,174). coords(br1gt_2,86,207,174).  visible(br1gt_2,174). happensAt(inactive(br1gt_2),174).
orientation(br1gt_0,33,175). coords(br1gt_0,292,256,175).  visible(br1gt_0,175). happensAt(walking(br1gt_0),175). orientation(br1gt_2,171,175). coords(br1gt_2,86,207,175).  visible(br1gt_2,175). happensAt(inactive(br1gt_2),175).
orientation(br1gt_0,39,176). coords(br1gt_0,293,258,176).  visible(br1gt_0,176). happensAt(walking(br1gt_0),176). orientation(br1gt_2,171,176). coords(br1gt_2,86,207,176).  visible(br1gt_2,176). happensAt(inactive(br1gt_2),176).
orientation(br1gt_0,39,177). coords(br1gt_0,294,260,177).  visible(br1gt_0,177). happensAt(walking(br1gt_0),177). orientation(br1gt_2,171,177). coords(br1gt_2,86,207,177).  visible(br1gt_2,177). happensAt(inactive(br1gt_2),177).
orientation(br1gt_0,39,178). coords(br1gt_0,295,263,178).  visible(br1gt_0,178). happensAt(walking(br1gt_0),178). orientation(br1gt_2,171,178). coords(br1gt_2,86,207,178).  visible(br1gt_2,178). happensAt(inactive(br1gt_2),178).
orientation(br1gt_0,50,179). coords(br1gt_0,296,266,179).  visible(br1gt_0,179). happensAt(walking(br1gt_0),179). orientation(br1gt_2,171,179). coords(br1gt_2,86,207,179).  visible(br1gt_2,179). happensAt(inactive(br1gt_2),179).
orientation(br1gt_0,50,180). coords(br1gt_0,298,268,180).  visible(br1gt_0,180). happensAt(walking(br1gt_0),180). orientation(br1gt_2,171,180). coords(br1gt_2,86,207,180).  visible(br1gt_2,180). happensAt(inactive(br1gt_2),180).
#show holdsAt/2.