
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

	orientation(rsfgt_0,160,156). coords(rsfgt_0,34,155,156).  visible(rsfgt_0,156). happensAt(walking(rsfgt_0),156). orientation(rsfgt_1,150,156). coords(rsfgt_1,179,131,156).  visible(rsfgt_1,156). happensAt(walking(rsfgt_1),156).
orientation(rsfgt_0,160,157). coords(rsfgt_0,34,155,157).  visible(rsfgt_0,157). happensAt(walking(rsfgt_0),157). orientation(rsfgt_1,150,157). coords(rsfgt_1,179,131,157).  visible(rsfgt_1,157). happensAt(walking(rsfgt_1),157).
orientation(rsfgt_0,160,158). coords(rsfgt_0,34,155,158).  visible(rsfgt_0,158). happensAt(walking(rsfgt_0),158). orientation(rsfgt_1,150,158). coords(rsfgt_1,181,133,158).  visible(rsfgt_1,158). happensAt(walking(rsfgt_1),158).
orientation(rsfgt_0,160,159). coords(rsfgt_0,34,157,159).  visible(rsfgt_0,159). happensAt(walking(rsfgt_0),159). orientation(rsfgt_1,150,159). coords(rsfgt_1,181,133,159).  visible(rsfgt_1,159). happensAt(walking(rsfgt_1),159).
orientation(rsfgt_0,160,160). coords(rsfgt_0,34,157,160).  visible(rsfgt_0,160). happensAt(walking(rsfgt_0),160). orientation(rsfgt_1,150,160). coords(rsfgt_1,183,135,160).  visible(rsfgt_1,160). happensAt(walking(rsfgt_1),160).
orientation(rsfgt_0,160,161). coords(rsfgt_0,32,157,161).  visible(rsfgt_0,161). happensAt(walking(rsfgt_0),161). orientation(rsfgt_1,150,161). coords(rsfgt_1,186,137,161).  visible(rsfgt_1,161). happensAt(walking(rsfgt_1),161).
orientation(rsfgt_0,160,162). coords(rsfgt_0,32,157,162).  visible(rsfgt_0,162). happensAt(walking(rsfgt_0),162). orientation(rsfgt_1,150,162). coords(rsfgt_1,189,138,162).  visible(rsfgt_1,162). happensAt(walking(rsfgt_1),162).
orientation(rsfgt_0,160,163). coords(rsfgt_0,32,157,163).  visible(rsfgt_0,163). happensAt(walking(rsfgt_0),163). orientation(rsfgt_1,150,163). coords(rsfgt_1,192,141,163).  visible(rsfgt_1,163). happensAt(walking(rsfgt_1),163).
orientation(rsfgt_0,160,164). coords(rsfgt_0,32,157,164).  visible(rsfgt_0,164). happensAt(walking(rsfgt_0),164). orientation(rsfgt_1,150,164). coords(rsfgt_1,194,143,164).  visible(rsfgt_1,164). happensAt(walking(rsfgt_1),164).
orientation(rsfgt_0,160,165). coords(rsfgt_0,32,157,165).  visible(rsfgt_0,165). happensAt(walking(rsfgt_0),165). orientation(rsfgt_1,150,165). coords(rsfgt_1,195,144,165).  visible(rsfgt_1,165). happensAt(walking(rsfgt_1),165).
orientation(rsfgt_0,160,166). coords(rsfgt_0,31,157,166).  visible(rsfgt_0,166). happensAt(walking(rsfgt_0),166). orientation(rsfgt_1,150,166). coords(rsfgt_1,196,145,166).  visible(rsfgt_1,166). happensAt(walking(rsfgt_1),166).
orientation(rsfgt_0,160,167). coords(rsfgt_0,30,157,167).  visible(rsfgt_0,167). happensAt(walking(rsfgt_0),167). orientation(rsfgt_1,150,167). coords(rsfgt_1,198,147,167).  visible(rsfgt_1,167). happensAt(walking(rsfgt_1),167).
orientation(rsfgt_0,160,168). coords(rsfgt_0,30,159,168).  visible(rsfgt_0,168). happensAt(walking(rsfgt_0),168). orientation(rsfgt_1,150,168). coords(rsfgt_1,199,148,168).  visible(rsfgt_1,168). happensAt(walking(rsfgt_1),168).
orientation(rsfgt_0,160,169). coords(rsfgt_0,30,159,169).  visible(rsfgt_0,169). happensAt(walking(rsfgt_0),169). orientation(rsfgt_1,150,169). coords(rsfgt_1,199,149,169).  visible(rsfgt_1,169). happensAt(walking(rsfgt_1),169).
orientation(rsfgt_0,160,170). coords(rsfgt_0,30,161,170).  visible(rsfgt_0,170). happensAt(walking(rsfgt_0),170). orientation(rsfgt_1,150,170). coords(rsfgt_1,199,149,170).  visible(rsfgt_1,170). happensAt(walking(rsfgt_1),170).
orientation(rsfgt_0,160,171). coords(rsfgt_0,30,161,171).  visible(rsfgt_0,171). happensAt(walking(rsfgt_0),171). orientation(rsfgt_1,150,171). coords(rsfgt_1,199,149,171).  visible(rsfgt_1,171). happensAt(walking(rsfgt_1),171).
orientation(rsfgt_0,160,172). coords(rsfgt_0,30,161,172).  visible(rsfgt_0,172). happensAt(walking(rsfgt_0),172). orientation(rsfgt_1,150,172). coords(rsfgt_1,199,149,172).  visible(rsfgt_1,172). happensAt(walking(rsfgt_1),172).
orientation(rsfgt_0,160,173). coords(rsfgt_0,30,161,173).  visible(rsfgt_0,173). happensAt(walking(rsfgt_0),173). orientation(rsfgt_1,150,173). coords(rsfgt_1,201,153,173).  visible(rsfgt_1,173). happensAt(walking(rsfgt_1),173).
orientation(rsfgt_0,160,174). coords(rsfgt_0,30,161,174).  visible(rsfgt_0,174). happensAt(walking(rsfgt_0),174). orientation(rsfgt_1,150,174). coords(rsfgt_1,201,157,174).  visible(rsfgt_1,174). happensAt(walking(rsfgt_1),174).
orientation(rsfgt_0,160,175). coords(rsfgt_0,30,161,175).  visible(rsfgt_0,175). happensAt(walking(rsfgt_0),175). orientation(rsfgt_1,150,175). coords(rsfgt_1,203,160,175).  visible(rsfgt_1,175). happensAt(walking(rsfgt_1),175).
orientation(rsfgt_0,160,176). coords(rsfgt_0,30,161,176).  visible(rsfgt_0,176). happensAt(walking(rsfgt_0),176). orientation(rsfgt_1,150,176). coords(rsfgt_1,206,163,176).  visible(rsfgt_1,176). happensAt(walking(rsfgt_1),176).
orientation(rsfgt_0,160,177). coords(rsfgt_0,30,162,177).  visible(rsfgt_0,177). happensAt(walking(rsfgt_0),177). orientation(rsfgt_1,150,177). coords(rsfgt_1,209,167,177).  visible(rsfgt_1,177). happensAt(walking(rsfgt_1),177).
orientation(rsfgt_0,160,178). coords(rsfgt_0,26,164,178).  visible(rsfgt_0,178). happensAt(walking(rsfgt_0),178). orientation(rsfgt_1,150,178). coords(rsfgt_1,211,169,178).  visible(rsfgt_1,178). happensAt(walking(rsfgt_1),178).
orientation(rsfgt_0,160,179). coords(rsfgt_0,26,164,179).  visible(rsfgt_0,179). happensAt(walking(rsfgt_0),179). orientation(rsfgt_1,150,179). coords(rsfgt_1,213,170,179).  visible(rsfgt_1,179). happensAt(walking(rsfgt_1),179).
orientation(rsfgt_0,160,180). coords(rsfgt_0,26,164,180).  visible(rsfgt_0,180). happensAt(walking(rsfgt_0),180). orientation(rsfgt_1,150,180). coords(rsfgt_1,215,172,180).  visible(rsfgt_1,180). happensAt(walking(rsfgt_1),180).
#show holdsAt/2.