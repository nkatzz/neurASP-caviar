
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

	orientation(lbgt_0,161,156). coords(lbgt_0,211,199,156).  visible(lbgt_0,156). happensAt(walking(lbgt_0),156). orientation(lbgt_1,157,156). coords(lbgt_1,79,94,156).  visible(lbgt_1,156). happensAt(walking(lbgt_1),156).
orientation(lbgt_0,161,157). coords(lbgt_0,213,200,157).  visible(lbgt_0,157). happensAt(walking(lbgt_0),157). orientation(lbgt_1,157,157). coords(lbgt_1,79,94,157).  visible(lbgt_1,157). happensAt(walking(lbgt_1),157).
orientation(lbgt_0,165,158). coords(lbgt_0,213,201,158).  visible(lbgt_0,158). happensAt(walking(lbgt_0),158). orientation(lbgt_1,157,158). coords(lbgt_1,79,94,158).  visible(lbgt_1,158). happensAt(walking(lbgt_1),158).
orientation(lbgt_0,165,159). coords(lbgt_0,214,201,159).  visible(lbgt_0,159). happensAt(active(lbgt_0),159). orientation(lbgt_1,157,159). coords(lbgt_1,79,94,159).  visible(lbgt_1,159). happensAt(walking(lbgt_1),159).
orientation(lbgt_0,165,160). coords(lbgt_0,212,202,160).  visible(lbgt_0,160). happensAt(active(lbgt_0),160). orientation(lbgt_1,157,160). coords(lbgt_1,79,95,160).  visible(lbgt_1,160). happensAt(walking(lbgt_1),160).
orientation(lbgt_0,165,161). coords(lbgt_0,211,202,161).  visible(lbgt_0,161). happensAt(active(lbgt_0),161). orientation(lbgt_1,157,161). coords(lbgt_1,79,97,161).  visible(lbgt_1,161). happensAt(walking(lbgt_1),161).
orientation(lbgt_0,168,162). coords(lbgt_0,209,201,162).  visible(lbgt_0,162). happensAt(active(lbgt_0),162). orientation(lbgt_1,157,162). coords(lbgt_1,79,97,162).  visible(lbgt_1,162). happensAt(walking(lbgt_1),162).
orientation(lbgt_0,168,163). coords(lbgt_0,207,201,163).  visible(lbgt_0,163). happensAt(active(lbgt_0),163). orientation(lbgt_1,157,163). coords(lbgt_1,79,97,163).  visible(lbgt_1,163). happensAt(walking(lbgt_1),163).
orientation(lbgt_0,168,164). coords(lbgt_0,205,201,164).  visible(lbgt_0,164). happensAt(active(lbgt_0),164). orientation(lbgt_1,152,164). coords(lbgt_1,79,97,164).  visible(lbgt_1,164). happensAt(walking(lbgt_1),164).
orientation(lbgt_0,168,165). coords(lbgt_0,205,201,165).  visible(lbgt_0,165). happensAt(active(lbgt_0),165). orientation(lbgt_1,152,165). coords(lbgt_1,79,98,165).  visible(lbgt_1,165). happensAt(walking(lbgt_1),165).
orientation(lbgt_0,168,166). coords(lbgt_0,205,201,166).  visible(lbgt_0,166). happensAt(active(lbgt_0),166). orientation(lbgt_1,152,166). coords(lbgt_1,79,98,166).  visible(lbgt_1,166). happensAt(walking(lbgt_1),166).
orientation(lbgt_0,168,167). coords(lbgt_0,206,202,167).  visible(lbgt_0,167). happensAt(active(lbgt_0),167). orientation(lbgt_1,152,167). coords(lbgt_1,79,100,167).  visible(lbgt_1,167). happensAt(walking(lbgt_1),167).
orientation(lbgt_0,168,168). coords(lbgt_0,206,202,168).  visible(lbgt_0,168). happensAt(active(lbgt_0),168). orientation(lbgt_1,152,168). coords(lbgt_1,79,98,168).  visible(lbgt_1,168). happensAt(walking(lbgt_1),168).
orientation(lbgt_0,168,169). coords(lbgt_0,206,202,169).  visible(lbgt_0,169). happensAt(active(lbgt_0),169). orientation(lbgt_1,152,169). coords(lbgt_1,79,98,169).  visible(lbgt_1,169). happensAt(walking(lbgt_1),169).
orientation(lbgt_0,168,170). coords(lbgt_0,206,201,170).  visible(lbgt_0,170). happensAt(active(lbgt_0),170). orientation(lbgt_1,152,170). coords(lbgt_1,79,98,170).  visible(lbgt_1,170). happensAt(walking(lbgt_1),170).
orientation(lbgt_0,0,171). coords(lbgt_0,206,200,171).  visible(lbgt_0,171). happensAt(active(lbgt_0),171). orientation(lbgt_1,152,171). coords(lbgt_1,79,98,171).  visible(lbgt_1,171). happensAt(walking(lbgt_1),171).
orientation(lbgt_0,0,172). coords(lbgt_0,206,197,172).  visible(lbgt_0,172). happensAt(active(lbgt_0),172). orientation(lbgt_1,152,172). coords(lbgt_1,79,98,172).  visible(lbgt_1,172). happensAt(walking(lbgt_1),172).
orientation(lbgt_0,0,173). coords(lbgt_0,207,198,173).  visible(lbgt_0,173). happensAt(active(lbgt_0),173). orientation(lbgt_1,152,173). coords(lbgt_1,79,98,173).  visible(lbgt_1,173). happensAt(walking(lbgt_1),173).
orientation(lbgt_0,0,174). coords(lbgt_0,208,197,174).  visible(lbgt_0,174). happensAt(active(lbgt_0),174). orientation(lbgt_1,152,174). coords(lbgt_1,79,98,174).  visible(lbgt_1,174). happensAt(walking(lbgt_1),174).
orientation(lbgt_0,0,175). coords(lbgt_0,208,197,175).  visible(lbgt_0,175). happensAt(active(lbgt_0),175). orientation(lbgt_1,152,175). coords(lbgt_1,79,99,175).  visible(lbgt_1,175). happensAt(walking(lbgt_1),175).
orientation(lbgt_0,0,176). coords(lbgt_0,211,196,176).  visible(lbgt_0,176). happensAt(active(lbgt_0),176). orientation(lbgt_1,152,176). coords(lbgt_1,79,98,176).  visible(lbgt_1,176). happensAt(walking(lbgt_1),176).
orientation(lbgt_0,0,177). coords(lbgt_0,213,196,177).  visible(lbgt_0,177). happensAt(active(lbgt_0),177). orientation(lbgt_1,152,177). coords(lbgt_1,79,99,177).  visible(lbgt_1,177). happensAt(walking(lbgt_1),177).
orientation(lbgt_0,176,178). coords(lbgt_0,214,195,178).  visible(lbgt_0,178). happensAt(walking(lbgt_0),178). orientation(lbgt_1,152,178). coords(lbgt_1,79,98,178).  visible(lbgt_1,178). happensAt(walking(lbgt_1),178).
orientation(lbgt_0,171,179). coords(lbgt_0,214,195,179).  visible(lbgt_0,179). happensAt(walking(lbgt_0),179). orientation(lbgt_1,152,179). coords(lbgt_1,79,98,179).  visible(lbgt_1,179). happensAt(walking(lbgt_1),179).
orientation(lbgt_0,171,180). coords(lbgt_0,213,194,180).  visible(lbgt_0,180). happensAt(walking(lbgt_0),180). orientation(lbgt_1,152,180). coords(lbgt_1,79,98,180).  visible(lbgt_1,180). happensAt(walking(lbgt_1),180).
#show holdsAt/2.