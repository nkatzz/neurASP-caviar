
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

	orientation(fcgt_3,159,156). coords(fcgt_3,103,163,156).  visible(fcgt_3,156). happensAt(inactive(fcgt_3),156). orientation(fcgt_4,170,156). coords(fcgt_4,100,210,156).  visible(fcgt_4,156). happensAt(walking(fcgt_4),156).
orientation(fcgt_3,159,157). coords(fcgt_3,103,163,157).  visible(fcgt_3,157). happensAt(inactive(fcgt_3),157). orientation(fcgt_4,170,157). coords(fcgt_4,100,210,157).  visible(fcgt_4,157). happensAt(walking(fcgt_4),157).
orientation(fcgt_3,159,158). coords(fcgt_3,103,163,158).  visible(fcgt_3,158). happensAt(inactive(fcgt_3),158). orientation(fcgt_4,170,158). coords(fcgt_4,100,210,158).  visible(fcgt_4,158). happensAt(walking(fcgt_4),158).
orientation(fcgt_3,159,159). coords(fcgt_3,103,163,159).  visible(fcgt_3,159). happensAt(inactive(fcgt_3),159). orientation(fcgt_4,170,159). coords(fcgt_4,100,210,159).  visible(fcgt_4,159). happensAt(walking(fcgt_4),159).
orientation(fcgt_3,159,160). coords(fcgt_3,103,163,160).  visible(fcgt_3,160). happensAt(inactive(fcgt_3),160). orientation(fcgt_4,170,160). coords(fcgt_4,100,210,160).  visible(fcgt_4,160). happensAt(walking(fcgt_4),160).
orientation(fcgt_3,159,161). coords(fcgt_3,103,163,161).  visible(fcgt_3,161). happensAt(inactive(fcgt_3),161). orientation(fcgt_4,170,161). coords(fcgt_4,100,210,161).  visible(fcgt_4,161). happensAt(walking(fcgt_4),161).
orientation(fcgt_3,159,162). coords(fcgt_3,103,163,162).  visible(fcgt_3,162). happensAt(inactive(fcgt_3),162). orientation(fcgt_4,165,162). coords(fcgt_4,99,210,162).  visible(fcgt_4,162). happensAt(walking(fcgt_4),162).
orientation(fcgt_3,159,163). coords(fcgt_3,103,163,163).  visible(fcgt_3,163). happensAt(inactive(fcgt_3),163). orientation(fcgt_4,165,163). coords(fcgt_4,99,210,163).  visible(fcgt_4,163). happensAt(walking(fcgt_4),163).
orientation(fcgt_3,159,164). coords(fcgt_3,103,163,164).  visible(fcgt_3,164). happensAt(inactive(fcgt_3),164). orientation(fcgt_4,165,164). coords(fcgt_4,99,210,164).  visible(fcgt_4,164). happensAt(walking(fcgt_4),164).
orientation(fcgt_3,159,165). coords(fcgt_3,103,163,165).  visible(fcgt_3,165). happensAt(inactive(fcgt_3),165). orientation(fcgt_4,164,165). coords(fcgt_4,99,209,165).  visible(fcgt_4,165). happensAt(walking(fcgt_4),165).
orientation(fcgt_3,159,166). coords(fcgt_3,103,163,166).  visible(fcgt_3,166). happensAt(inactive(fcgt_3),166). orientation(fcgt_4,165,166). coords(fcgt_4,99,209,166).  visible(fcgt_4,166). happensAt(walking(fcgt_4),166).
orientation(fcgt_3,159,167). coords(fcgt_3,103,163,167).  visible(fcgt_3,167). happensAt(inactive(fcgt_3),167). orientation(fcgt_4,165,167). coords(fcgt_4,99,209,167).  visible(fcgt_4,167). happensAt(walking(fcgt_4),167).
orientation(fcgt_3,159,168). coords(fcgt_3,103,163,168).  visible(fcgt_3,168). happensAt(inactive(fcgt_3),168). orientation(fcgt_4,163,168). coords(fcgt_4,99,209,168).  visible(fcgt_4,168). happensAt(walking(fcgt_4),168).
orientation(fcgt_3,159,169). coords(fcgt_3,103,163,169).  visible(fcgt_3,169). happensAt(inactive(fcgt_3),169). orientation(fcgt_4,163,169). coords(fcgt_4,98,208,169).  visible(fcgt_4,169). happensAt(walking(fcgt_4),169).
orientation(fcgt_3,159,170). coords(fcgt_3,103,163,170).  visible(fcgt_3,170). happensAt(inactive(fcgt_3),170). orientation(fcgt_4,165,170). coords(fcgt_4,98,208,170).  visible(fcgt_4,170). happensAt(walking(fcgt_4),170).
orientation(fcgt_3,159,171). coords(fcgt_3,103,163,171).  visible(fcgt_3,171). happensAt(inactive(fcgt_3),171). orientation(fcgt_4,162,171). coords(fcgt_4,98,208,171).  visible(fcgt_4,171). happensAt(walking(fcgt_4),171).
orientation(fcgt_3,159,172). coords(fcgt_3,103,163,172).  visible(fcgt_3,172). happensAt(inactive(fcgt_3),172). orientation(fcgt_4,163,172). coords(fcgt_4,97,208,172).  visible(fcgt_4,172). happensAt(walking(fcgt_4),172).
orientation(fcgt_3,159,173). coords(fcgt_3,103,163,173).  visible(fcgt_3,173). happensAt(inactive(fcgt_3),173). orientation(fcgt_4,163,173). coords(fcgt_4,96,208,173).  visible(fcgt_4,173). happensAt(walking(fcgt_4),173).
orientation(fcgt_3,159,174). coords(fcgt_3,103,163,174).  visible(fcgt_3,174). happensAt(inactive(fcgt_3),174). orientation(fcgt_4,163,174). coords(fcgt_4,96,207,174).  visible(fcgt_4,174). happensAt(walking(fcgt_4),174).
orientation(fcgt_3,159,175). coords(fcgt_3,103,163,175).  visible(fcgt_3,175). happensAt(inactive(fcgt_3),175). orientation(fcgt_4,163,175). coords(fcgt_4,95,206,175).  visible(fcgt_4,175). happensAt(walking(fcgt_4),175).
orientation(fcgt_3,159,176). coords(fcgt_3,103,163,176).  visible(fcgt_3,176). happensAt(inactive(fcgt_3),176). orientation(fcgt_4,163,176). coords(fcgt_4,94,206,176).  visible(fcgt_4,176). happensAt(walking(fcgt_4),176).
orientation(fcgt_3,159,177). coords(fcgt_3,103,163,177).  visible(fcgt_3,177). happensAt(inactive(fcgt_3),177). orientation(fcgt_4,163,177). coords(fcgt_4,94,206,177).  visible(fcgt_4,177). happensAt(walking(fcgt_4),177).
orientation(fcgt_3,159,178). coords(fcgt_3,103,163,178).  visible(fcgt_3,178). happensAt(inactive(fcgt_3),178). orientation(fcgt_4,163,178). coords(fcgt_4,93,205,178).  visible(fcgt_4,178). happensAt(walking(fcgt_4),178).
orientation(fcgt_3,159,179). coords(fcgt_3,103,163,179).  visible(fcgt_3,179). happensAt(inactive(fcgt_3),179). orientation(fcgt_4,163,179). coords(fcgt_4,93,205,179).  visible(fcgt_4,179). happensAt(walking(fcgt_4),179).
orientation(fcgt_3,159,180). coords(fcgt_3,103,163,180).  visible(fcgt_3,180). happensAt(inactive(fcgt_3),180). orientation(fcgt_4,163,180). coords(fcgt_4,93,205,180).  visible(fcgt_4,180). happensAt(walking(fcgt_4),180).
#show holdsAt/2.