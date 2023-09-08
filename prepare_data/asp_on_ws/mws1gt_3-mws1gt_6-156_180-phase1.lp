
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

	orientation(mws1gt_3,154,156). coords(mws1gt_3,173,154,156).  visible(mws1gt_3,156). happensAt(walking(mws1gt_3),156). orientation(mws1gt_6,157,156). coords(mws1gt_6,73,56,156).  happensAt(appear(mws1gt_6),156). happensAt(walking(mws1gt_6),156).
orientation(mws1gt_3,154,157). coords(mws1gt_3,171,154,157).  visible(mws1gt_3,157). happensAt(walking(mws1gt_3),157). orientation(mws1gt_6,157,157). coords(mws1gt_6,74,57,157).  visible(mws1gt_6,157). happensAt(walking(mws1gt_6),157).
orientation(mws1gt_3,154,158). coords(mws1gt_3,171,154,158).  visible(mws1gt_3,158). happensAt(walking(mws1gt_3),158). orientation(mws1gt_6,146,158). coords(mws1gt_6,74,57,158).  visible(mws1gt_6,158). happensAt(walking(mws1gt_6),158).
orientation(mws1gt_3,154,159). coords(mws1gt_3,171,154,159).  visible(mws1gt_3,159). happensAt(walking(mws1gt_3),159). orientation(mws1gt_6,146,159). coords(mws1gt_6,74,58,159).  visible(mws1gt_6,159). happensAt(walking(mws1gt_6),159).
orientation(mws1gt_3,154,160). coords(mws1gt_3,170,154,160).  visible(mws1gt_3,160). happensAt(walking(mws1gt_3),160). orientation(mws1gt_6,146,160). coords(mws1gt_6,73,58,160).  visible(mws1gt_6,160). happensAt(walking(mws1gt_6),160).
orientation(mws1gt_3,154,161). coords(mws1gt_3,170,154,161).  visible(mws1gt_3,161). happensAt(walking(mws1gt_3),161). orientation(mws1gt_6,146,161). coords(mws1gt_6,74,61,161).  visible(mws1gt_6,161). happensAt(walking(mws1gt_6),161).
orientation(mws1gt_3,154,162). coords(mws1gt_3,169,154,162).  visible(mws1gt_3,162). happensAt(walking(mws1gt_3),162). orientation(mws1gt_6,146,162). coords(mws1gt_6,74,62,162).  visible(mws1gt_6,162). happensAt(walking(mws1gt_6),162).
orientation(mws1gt_3,154,163). coords(mws1gt_3,169,154,163).  visible(mws1gt_3,163). happensAt(walking(mws1gt_3),163). orientation(mws1gt_6,146,163). coords(mws1gt_6,74,62,163).  visible(mws1gt_6,163). happensAt(walking(mws1gt_6),163).
orientation(mws1gt_3,154,164). coords(mws1gt_3,169,154,164).  visible(mws1gt_3,164). happensAt(walking(mws1gt_3),164). orientation(mws1gt_6,146,164). coords(mws1gt_6,74,62,164).  visible(mws1gt_6,164). happensAt(walking(mws1gt_6),164).
orientation(mws1gt_3,154,165). coords(mws1gt_3,168,154,165).  visible(mws1gt_3,165). happensAt(walking(mws1gt_3),165). orientation(mws1gt_6,146,165). coords(mws1gt_6,74,65,165).  visible(mws1gt_6,165). happensAt(walking(mws1gt_6),165).
orientation(mws1gt_3,154,166). coords(mws1gt_3,167,154,166).  visible(mws1gt_3,166). happensAt(walking(mws1gt_3),166). orientation(mws1gt_6,146,166). coords(mws1gt_6,74,65,166).  visible(mws1gt_6,166). happensAt(walking(mws1gt_6),166).
orientation(mws1gt_3,154,167). coords(mws1gt_3,166,154,167).  visible(mws1gt_3,167). happensAt(walking(mws1gt_3),167). orientation(mws1gt_6,146,167). coords(mws1gt_6,73,65,167).  visible(mws1gt_6,167). happensAt(walking(mws1gt_6),167).
orientation(mws1gt_3,154,168). coords(mws1gt_3,164,154,168).  visible(mws1gt_3,168). happensAt(walking(mws1gt_3),168). orientation(mws1gt_6,146,168). coords(mws1gt_6,73,65,168).  visible(mws1gt_6,168). happensAt(walking(mws1gt_6),168).
orientation(mws1gt_3,154,169). coords(mws1gt_3,163,155,169).  visible(mws1gt_3,169). happensAt(walking(mws1gt_3),169). orientation(mws1gt_6,146,169). coords(mws1gt_6,73,66,169).  visible(mws1gt_6,169). happensAt(walking(mws1gt_6),169).
orientation(mws1gt_3,154,170). coords(mws1gt_3,162,155,170).  visible(mws1gt_3,170). happensAt(walking(mws1gt_3),170). orientation(mws1gt_6,146,170). coords(mws1gt_6,73,67,170).  visible(mws1gt_6,170). happensAt(walking(mws1gt_6),170).
orientation(mws1gt_3,154,171). coords(mws1gt_3,160,154,171).  visible(mws1gt_3,171). happensAt(walking(mws1gt_3),171). orientation(mws1gt_6,146,171). coords(mws1gt_6,72,67,171).  visible(mws1gt_6,171). happensAt(walking(mws1gt_6),171).
orientation(mws1gt_3,154,172). coords(mws1gt_3,159,154,172).  visible(mws1gt_3,172). happensAt(walking(mws1gt_3),172). orientation(mws1gt_6,146,172). coords(mws1gt_6,72,67,172).  visible(mws1gt_6,172). happensAt(walking(mws1gt_6),172).
orientation(mws1gt_3,154,173). coords(mws1gt_3,159,154,173).  visible(mws1gt_3,173). happensAt(walking(mws1gt_3),173). orientation(mws1gt_6,146,173). coords(mws1gt_6,72,68,173).  visible(mws1gt_6,173). happensAt(walking(mws1gt_6),173).
orientation(mws1gt_3,154,174). coords(mws1gt_3,158,154,174).  visible(mws1gt_3,174). happensAt(walking(mws1gt_3),174). orientation(mws1gt_6,140,174). coords(mws1gt_6,72,68,174).  visible(mws1gt_6,174). happensAt(walking(mws1gt_6),174).
orientation(mws1gt_3,154,175). coords(mws1gt_3,158,154,175).  visible(mws1gt_3,175). happensAt(walking(mws1gt_3),175). orientation(mws1gt_6,140,175). coords(mws1gt_6,71,71,175).  visible(mws1gt_6,175). happensAt(walking(mws1gt_6),175).
orientation(mws1gt_3,154,176). coords(mws1gt_3,157,154,176).  visible(mws1gt_3,176). happensAt(walking(mws1gt_3),176). orientation(mws1gt_6,140,176). coords(mws1gt_6,70,71,176).  visible(mws1gt_6,176). happensAt(walking(mws1gt_6),176).
orientation(mws1gt_3,154,177). coords(mws1gt_3,157,154,177).  visible(mws1gt_3,177). happensAt(walking(mws1gt_3),177). orientation(mws1gt_6,140,177). coords(mws1gt_6,70,72,177).  visible(mws1gt_6,177). happensAt(walking(mws1gt_6),177).
orientation(mws1gt_3,154,178). coords(mws1gt_3,156,154,178).  visible(mws1gt_3,178). happensAt(walking(mws1gt_3),178). orientation(mws1gt_6,140,178). coords(mws1gt_6,70,73,178).  visible(mws1gt_6,178). happensAt(walking(mws1gt_6),178).
orientation(mws1gt_3,154,179). coords(mws1gt_3,156,154,179).  visible(mws1gt_3,179). happensAt(walking(mws1gt_3),179). orientation(mws1gt_6,140,179). coords(mws1gt_6,70,73,179).  visible(mws1gt_6,179). happensAt(walking(mws1gt_6),179).
orientation(mws1gt_3,154,180). coords(mws1gt_3,156,153,180).  visible(mws1gt_3,180). happensAt(walking(mws1gt_3),180). orientation(mws1gt_6,140,180). coords(mws1gt_6,70,73,180).  visible(mws1gt_6,180). happensAt(walking(mws1gt_6),180).
#show holdsAt/2.