
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

	orientation(spgt_0,154,156). coords(spgt_0,107,123,156).  visible(spgt_0,156). happensAt(walking(spgt_0),156). orientation(spgt_1,148,156). coords(spgt_1,103,109,156).  visible(spgt_1,156). happensAt(walking(spgt_1),156).
orientation(spgt_0,154,157). coords(spgt_0,108,123,157).  visible(spgt_0,157). happensAt(walking(spgt_0),157). orientation(spgt_1,148,157). coords(spgt_1,104,109,157).  visible(spgt_1,157). happensAt(walking(spgt_1),157).
orientation(spgt_0,150,158). coords(spgt_0,108,123,158).  visible(spgt_0,158). happensAt(walking(spgt_0),158). orientation(spgt_1,148,158). coords(spgt_1,105,109,158).  visible(spgt_1,158). happensAt(walking(spgt_1),158).
orientation(spgt_0,150,159). coords(spgt_0,109,122,159).  visible(spgt_0,159). happensAt(walking(spgt_0),159). orientation(spgt_1,148,159). coords(spgt_1,106,108,159).  visible(spgt_1,159). happensAt(walking(spgt_1),159).
orientation(spgt_0,148,160). coords(spgt_0,109,122,160).  visible(spgt_0,160). happensAt(walking(spgt_0),160). orientation(spgt_1,148,160). coords(spgt_1,107,108,160).  visible(spgt_1,160). happensAt(walking(spgt_1),160).
orientation(spgt_0,148,161). coords(spgt_0,109,121,161).  visible(spgt_0,161). happensAt(walking(spgt_0),161). orientation(spgt_1,148,161). coords(spgt_1,107,108,161).  visible(spgt_1,161). happensAt(walking(spgt_1),161).
orientation(spgt_0,148,162). coords(spgt_0,109,121,162).  visible(spgt_0,162). happensAt(walking(spgt_0),162). orientation(spgt_1,148,162). coords(spgt_1,107,108,162).  visible(spgt_1,162). happensAt(walking(spgt_1),162).
orientation(spgt_0,148,163). coords(spgt_0,109,120,163).  visible(spgt_0,163). happensAt(walking(spgt_0),163). orientation(spgt_1,148,163). coords(spgt_1,107,107,163).  visible(spgt_1,163). happensAt(active(spgt_1),163).
orientation(spgt_0,148,164). coords(spgt_0,109,120,164).  visible(spgt_0,164). happensAt(walking(spgt_0),164). orientation(spgt_1,148,164). coords(spgt_1,107,106,164).  visible(spgt_1,164). happensAt(active(spgt_1),164).
orientation(spgt_0,148,165). coords(spgt_0,109,120,165).  visible(spgt_0,165). happensAt(walking(spgt_0),165). orientation(spgt_1,148,165). coords(spgt_1,107,105,165).  visible(spgt_1,165). happensAt(active(spgt_1),165).
orientation(spgt_0,148,166). coords(spgt_0,109,118,166).  visible(spgt_0,166). happensAt(walking(spgt_0),166). orientation(spgt_1,148,166). coords(spgt_1,107,105,166).  visible(spgt_1,166). happensAt(active(spgt_1),166).
orientation(spgt_0,148,167). coords(spgt_0,110,118,167).  visible(spgt_0,167). happensAt(walking(spgt_0),167). orientation(spgt_1,148,167). coords(spgt_1,108,105,167).  visible(spgt_1,167). happensAt(active(spgt_1),167).
orientation(spgt_0,148,168). coords(spgt_0,111,116,168).  visible(spgt_0,168). happensAt(walking(spgt_0),168). orientation(spgt_1,148,168). coords(spgt_1,108,104,168).  visible(spgt_1,168). happensAt(active(spgt_1),168).
orientation(spgt_0,148,169). coords(spgt_0,112,116,169).  visible(spgt_0,169). happensAt(walking(spgt_0),169). orientation(spgt_1,148,169). coords(spgt_1,108,103,169).  visible(spgt_1,169). happensAt(active(spgt_1),169).
orientation(spgt_0,148,170). coords(spgt_0,112,116,170).  visible(spgt_0,170). happensAt(active(spgt_0),170). orientation(spgt_1,148,170). coords(spgt_1,108,103,170).  visible(spgt_1,170). happensAt(active(spgt_1),170).
orientation(spgt_0,148,171). coords(spgt_0,112,116,171).  visible(spgt_0,171). happensAt(active(spgt_0),171). orientation(spgt_1,148,171). coords(spgt_1,108,103,171).  visible(spgt_1,171). happensAt(active(spgt_1),171).
orientation(spgt_0,148,172). coords(spgt_0,112,116,172).  visible(spgt_0,172). happensAt(active(spgt_0),172). orientation(spgt_1,148,172). coords(spgt_1,108,103,172).  visible(spgt_1,172). happensAt(active(spgt_1),172).
orientation(spgt_0,148,173). coords(spgt_0,112,116,173).  visible(spgt_0,173). happensAt(active(spgt_0),173). orientation(spgt_1,148,173). coords(spgt_1,108,103,173).  visible(spgt_1,173). happensAt(active(spgt_1),173).
orientation(spgt_0,148,174). coords(spgt_0,113,115,174).  visible(spgt_0,174). happensAt(active(spgt_0),174). orientation(spgt_1,148,174). coords(spgt_1,108,103,174).  visible(spgt_1,174). happensAt(active(spgt_1),174).
orientation(spgt_0,148,175). coords(spgt_0,113,115,175).  visible(spgt_0,175). happensAt(active(spgt_0),175). orientation(spgt_1,148,175). coords(spgt_1,108,103,175).  visible(spgt_1,175). happensAt(active(spgt_1),175).
orientation(spgt_0,148,176). coords(spgt_0,113,114,176).  visible(spgt_0,176). happensAt(active(spgt_0),176). orientation(spgt_1,148,176). coords(spgt_1,108,103,176).  visible(spgt_1,176). happensAt(active(spgt_1),176).
orientation(spgt_0,148,177). coords(spgt_0,113,114,177).  visible(spgt_0,177). happensAt(active(spgt_0),177). orientation(spgt_1,148,177). coords(spgt_1,108,103,177).  visible(spgt_1,177). happensAt(active(spgt_1),177).
orientation(spgt_0,148,178). coords(spgt_0,113,114,178).  visible(spgt_0,178). happensAt(active(spgt_0),178). orientation(spgt_1,148,178). coords(spgt_1,108,103,178).  visible(spgt_1,178). happensAt(active(spgt_1),178).
orientation(spgt_0,148,179). coords(spgt_0,113,114,179).  visible(spgt_0,179). happensAt(active(spgt_0),179). orientation(spgt_1,148,179). coords(spgt_1,108,103,179).  visible(spgt_1,179). happensAt(active(spgt_1),179).
orientation(spgt_0,148,180). coords(spgt_0,113,114,180).  visible(spgt_0,180). happensAt(active(spgt_0),180). orientation(spgt_1,148,180). coords(spgt_1,108,103,180).  visible(spgt_1,180). happensAt(active(spgt_1),180).
#show holdsAt/2.