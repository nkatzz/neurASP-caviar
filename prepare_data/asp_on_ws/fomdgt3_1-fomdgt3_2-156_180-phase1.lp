
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

	orientation(fomdgt3_1,150,156). coords(fomdgt3_1,106,108,156).  visible(fomdgt3_1,156). happensAt(walking(fomdgt3_1),156). orientation(fomdgt3_2,18,156). coords(fomdgt3_2,260,221,156).  visible(fomdgt3_2,156). happensAt(walking(fomdgt3_2),156).
orientation(fomdgt3_1,150,157). coords(fomdgt3_1,99,108,157).  visible(fomdgt3_1,157). happensAt(walking(fomdgt3_1),157). orientation(fomdgt3_2,18,157). coords(fomdgt3_2,260,221,157).  visible(fomdgt3_2,157). happensAt(walking(fomdgt3_2),157).
orientation(fomdgt3_1,150,158). coords(fomdgt3_1,98,108,158).  visible(fomdgt3_1,158). happensAt(walking(fomdgt3_1),158). orientation(fomdgt3_2,170,158). coords(fomdgt3_2,260,218,158).  visible(fomdgt3_2,158). happensAt(walking(fomdgt3_2),158).
orientation(fomdgt3_1,150,159). coords(fomdgt3_1,98,108,159).  visible(fomdgt3_1,159). happensAt(walking(fomdgt3_1),159). orientation(fomdgt3_2,160,159). coords(fomdgt3_2,260,215,159).  visible(fomdgt3_2,159). happensAt(walking(fomdgt3_2),159).
orientation(fomdgt3_1,150,160). coords(fomdgt3_1,98,108,160).  visible(fomdgt3_1,160). happensAt(walking(fomdgt3_1),160). orientation(fomdgt3_2,160,160). coords(fomdgt3_2,257,214,160).  visible(fomdgt3_2,160). happensAt(walking(fomdgt3_2),160).
orientation(fomdgt3_1,150,161). coords(fomdgt3_1,97,108,161).  visible(fomdgt3_1,161). happensAt(walking(fomdgt3_1),161). orientation(fomdgt3_2,160,161). coords(fomdgt3_2,257,210,161).  visible(fomdgt3_2,161). happensAt(walking(fomdgt3_2),161).
orientation(fomdgt3_1,150,162). coords(fomdgt3_1,97,108,162).  visible(fomdgt3_1,162). happensAt(walking(fomdgt3_1),162). orientation(fomdgt3_2,160,162). coords(fomdgt3_2,254,207,162).  visible(fomdgt3_2,162). happensAt(walking(fomdgt3_2),162).
orientation(fomdgt3_1,150,163). coords(fomdgt3_1,97,108,163).  visible(fomdgt3_1,163). happensAt(walking(fomdgt3_1),163). orientation(fomdgt3_2,160,163). coords(fomdgt3_2,254,203,163).  visible(fomdgt3_2,163). happensAt(walking(fomdgt3_2),163).
orientation(fomdgt3_1,150,164). coords(fomdgt3_1,97,108,164).  visible(fomdgt3_1,164). happensAt(walking(fomdgt3_1),164). orientation(fomdgt3_2,160,164). coords(fomdgt3_2,253,200,164).  visible(fomdgt3_2,164). happensAt(walking(fomdgt3_2),164).
orientation(fomdgt3_1,150,165). coords(fomdgt3_1,97,108,165).  visible(fomdgt3_1,165). happensAt(walking(fomdgt3_1),165). orientation(fomdgt3_2,160,165). coords(fomdgt3_2,252,199,165).  visible(fomdgt3_2,165). happensAt(walking(fomdgt3_2),165).
orientation(fomdgt3_1,150,166). coords(fomdgt3_1,96,108,166).  visible(fomdgt3_1,166). happensAt(walking(fomdgt3_1),166). orientation(fomdgt3_2,160,166). coords(fomdgt3_2,252,196,166).  visible(fomdgt3_2,166). happensAt(walking(fomdgt3_2),166).
orientation(fomdgt3_1,150,167). coords(fomdgt3_1,96,108,167).  visible(fomdgt3_1,167). happensAt(walking(fomdgt3_1),167). orientation(fomdgt3_2,160,167). coords(fomdgt3_2,251,195,167).  visible(fomdgt3_2,167). happensAt(walking(fomdgt3_2),167).
orientation(fomdgt3_1,150,168). coords(fomdgt3_1,96,108,168).  visible(fomdgt3_1,168). happensAt(walking(fomdgt3_1),168). orientation(fomdgt3_2,160,168). coords(fomdgt3_2,251,195,168).  visible(fomdgt3_2,168). happensAt(walking(fomdgt3_2),168).
orientation(fomdgt3_1,150,169). coords(fomdgt3_1,92,107,169).  visible(fomdgt3_1,169). happensAt(walking(fomdgt3_1),169). orientation(fomdgt3_2,160,169). coords(fomdgt3_2,251,195,169).  visible(fomdgt3_2,169). happensAt(walking(fomdgt3_2),169).
orientation(fomdgt3_1,150,170). coords(fomdgt3_1,91,107,170).  visible(fomdgt3_1,170). happensAt(walking(fomdgt3_1),170). orientation(fomdgt3_2,160,170). coords(fomdgt3_2,250,193,170).  visible(fomdgt3_2,170). happensAt(walking(fomdgt3_2),170).
orientation(fomdgt3_1,150,171). coords(fomdgt3_1,90,106,171).  visible(fomdgt3_1,171). happensAt(walking(fomdgt3_1),171). orientation(fomdgt3_2,160,171). coords(fomdgt3_2,249,193,171).  visible(fomdgt3_2,171). happensAt(walking(fomdgt3_2),171).
orientation(fomdgt3_1,150,172). coords(fomdgt3_1,90,106,172).  visible(fomdgt3_1,172). happensAt(walking(fomdgt3_1),172). orientation(fomdgt3_2,160,172). coords(fomdgt3_2,249,192,172).  visible(fomdgt3_2,172). happensAt(walking(fomdgt3_2),172).
orientation(fomdgt3_1,150,173). coords(fomdgt3_1,89,105,173).  visible(fomdgt3_1,173). happensAt(walking(fomdgt3_1),173). orientation(fomdgt3_2,160,173). coords(fomdgt3_2,248,189,173).  visible(fomdgt3_2,173). happensAt(walking(fomdgt3_2),173).
orientation(fomdgt3_1,150,174). coords(fomdgt3_1,88,105,174).  visible(fomdgt3_1,174). happensAt(walking(fomdgt3_1),174). orientation(fomdgt3_2,160,174). coords(fomdgt3_2,247,187,174).  visible(fomdgt3_2,174). happensAt(walking(fomdgt3_2),174).
orientation(fomdgt3_1,150,175). coords(fomdgt3_1,89,105,175).  visible(fomdgt3_1,175). happensAt(walking(fomdgt3_1),175). orientation(fomdgt3_2,160,175). coords(fomdgt3_2,245,184,175).  visible(fomdgt3_2,175). happensAt(walking(fomdgt3_2),175).
orientation(fomdgt3_1,150,176). coords(fomdgt3_1,89,105,176).  visible(fomdgt3_1,176). happensAt(walking(fomdgt3_1),176). orientation(fomdgt3_2,160,176). coords(fomdgt3_2,245,184,176).  visible(fomdgt3_2,176). happensAt(walking(fomdgt3_2),176).
orientation(fomdgt3_1,150,177). coords(fomdgt3_1,89,105,177).  visible(fomdgt3_1,177). happensAt(walking(fomdgt3_1),177). orientation(fomdgt3_2,160,177). coords(fomdgt3_2,243,179,177).  visible(fomdgt3_2,177). happensAt(walking(fomdgt3_2),177).
orientation(fomdgt3_1,150,178). coords(fomdgt3_1,88,105,178).  visible(fomdgt3_1,178). happensAt(walking(fomdgt3_1),178). orientation(fomdgt3_2,160,178). coords(fomdgt3_2,242,177,178).  visible(fomdgt3_2,178). happensAt(walking(fomdgt3_2),178).
orientation(fomdgt3_1,150,179). coords(fomdgt3_1,89,105,179).  visible(fomdgt3_1,179). happensAt(walking(fomdgt3_1),179). orientation(fomdgt3_2,160,179). coords(fomdgt3_2,242,176,179).  visible(fomdgt3_2,179). happensAt(walking(fomdgt3_2),179).
orientation(fomdgt3_1,150,180). coords(fomdgt3_1,87,105,180).  visible(fomdgt3_1,180). happensAt(walking(fomdgt3_1),180). orientation(fomdgt3_2,160,180). coords(fomdgt3_2,242,176,180).  visible(fomdgt3_2,180). happensAt(walking(fomdgt3_2),180).
#show holdsAt/2.