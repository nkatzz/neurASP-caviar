
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

	orientation(wk2gt_1,135,72). coords(wk2gt_1,114,53,72).  visible(wk2gt_1,72). happensAt(active(wk2gt_1),72). orientation(wk2gt_2,141,72). coords(wk2gt_2,98,63,72).  visible(wk2gt_2,72). happensAt(active(wk2gt_2),72).
orientation(wk2gt_1,135,73). coords(wk2gt_1,114,53,73).  visible(wk2gt_1,73). happensAt(active(wk2gt_1),73). orientation(wk2gt_2,141,73). coords(wk2gt_2,98,63,73).  visible(wk2gt_2,73). happensAt(active(wk2gt_2),73).
orientation(wk2gt_1,138,74). coords(wk2gt_1,114,53,74).  visible(wk2gt_1,74). happensAt(active(wk2gt_1),74). orientation(wk2gt_2,141,74). coords(wk2gt_2,98,63,74).  visible(wk2gt_2,74). happensAt(active(wk2gt_2),74).
orientation(wk2gt_1,137,75). coords(wk2gt_1,114,53,75).  visible(wk2gt_1,75). happensAt(inactive(wk2gt_1),75). orientation(wk2gt_2,141,75). coords(wk2gt_2,98,63,75).  visible(wk2gt_2,75). happensAt(active(wk2gt_2),75).
orientation(wk2gt_1,137,76). coords(wk2gt_1,115,53,76).  visible(wk2gt_1,76). happensAt(inactive(wk2gt_1),76). orientation(wk2gt_2,141,76). coords(wk2gt_2,98,63,76).  visible(wk2gt_2,76). happensAt(active(wk2gt_2),76).
orientation(wk2gt_1,136,77). coords(wk2gt_1,115,53,77).  visible(wk2gt_1,77). happensAt(inactive(wk2gt_1),77). orientation(wk2gt_2,141,77). coords(wk2gt_2,98,63,77).  visible(wk2gt_2,77). happensAt(active(wk2gt_2),77).
orientation(wk2gt_1,141,78). coords(wk2gt_1,114,53,78).  visible(wk2gt_1,78). happensAt(inactive(wk2gt_1),78). orientation(wk2gt_2,141,78). coords(wk2gt_2,98,63,78).  visible(wk2gt_2,78). happensAt(active(wk2gt_2),78).
orientation(wk2gt_1,139,79). coords(wk2gt_1,114,53,79).  visible(wk2gt_1,79). happensAt(inactive(wk2gt_1),79). orientation(wk2gt_2,141,79). coords(wk2gt_2,98,63,79).  visible(wk2gt_2,79). happensAt(active(wk2gt_2),79).
orientation(wk2gt_1,135,80). coords(wk2gt_1,113,53,80).  visible(wk2gt_1,80). happensAt(inactive(wk2gt_1),80). orientation(wk2gt_2,141,80). coords(wk2gt_2,98,63,80).  visible(wk2gt_2,80). happensAt(active(wk2gt_2),80).
orientation(wk2gt_1,134,81). coords(wk2gt_1,113,53,81).  visible(wk2gt_1,81). happensAt(inactive(wk2gt_1),81). orientation(wk2gt_2,141,81). coords(wk2gt_2,98,63,81).  visible(wk2gt_2,81). happensAt(active(wk2gt_2),81).
orientation(wk2gt_1,134,82). coords(wk2gt_1,112,53,82).  visible(wk2gt_1,82). happensAt(inactive(wk2gt_1),82). orientation(wk2gt_2,141,82). coords(wk2gt_2,98,63,82).  visible(wk2gt_2,82). happensAt(active(wk2gt_2),82).
orientation(wk2gt_1,139,83). coords(wk2gt_1,112,53,83).  visible(wk2gt_1,83). happensAt(inactive(wk2gt_1),83). orientation(wk2gt_2,141,83). coords(wk2gt_2,98,63,83).  visible(wk2gt_2,83). happensAt(active(wk2gt_2),83).
orientation(wk2gt_1,137,84). coords(wk2gt_1,113,53,84).  visible(wk2gt_1,84). happensAt(inactive(wk2gt_1),84). orientation(wk2gt_2,141,84). coords(wk2gt_2,98,63,84).  visible(wk2gt_2,84). happensAt(active(wk2gt_2),84).
orientation(wk2gt_1,135,85). coords(wk2gt_1,114,53,85).  visible(wk2gt_1,85). happensAt(inactive(wk2gt_1),85). orientation(wk2gt_2,141,85). coords(wk2gt_2,98,63,85).  visible(wk2gt_2,85). happensAt(active(wk2gt_2),85).
orientation(wk2gt_1,136,86). coords(wk2gt_1,114,53,86).  visible(wk2gt_1,86). happensAt(inactive(wk2gt_1),86). orientation(wk2gt_2,141,86). coords(wk2gt_2,98,63,86).  visible(wk2gt_2,86). happensAt(active(wk2gt_2),86).
orientation(wk2gt_1,136,87). coords(wk2gt_1,114,53,87).  visible(wk2gt_1,87). happensAt(inactive(wk2gt_1),87). orientation(wk2gt_2,141,87). coords(wk2gt_2,98,63,87).  visible(wk2gt_2,87). happensAt(active(wk2gt_2),87).
orientation(wk2gt_1,136,88). coords(wk2gt_1,114,53,88).  visible(wk2gt_1,88). happensAt(inactive(wk2gt_1),88). orientation(wk2gt_2,141,88). coords(wk2gt_2,98,63,88).  visible(wk2gt_2,88). happensAt(active(wk2gt_2),88).
orientation(wk2gt_1,134,89). coords(wk2gt_1,114,53,89).  visible(wk2gt_1,89). happensAt(inactive(wk2gt_1),89). orientation(wk2gt_2,141,89). coords(wk2gt_2,98,63,89).  visible(wk2gt_2,89). happensAt(active(wk2gt_2),89).
orientation(wk2gt_1,138,90). coords(wk2gt_1,114,53,90).  visible(wk2gt_1,90). happensAt(inactive(wk2gt_1),90). orientation(wk2gt_2,141,90). coords(wk2gt_2,98,63,90).  visible(wk2gt_2,90). happensAt(active(wk2gt_2),90).
orientation(wk2gt_1,133,91). coords(wk2gt_1,115,53,91).  visible(wk2gt_1,91). happensAt(inactive(wk2gt_1),91). orientation(wk2gt_2,141,91). coords(wk2gt_2,98,62,91).  visible(wk2gt_2,91). happensAt(active(wk2gt_2),91).
orientation(wk2gt_1,135,92). coords(wk2gt_1,115,53,92).  visible(wk2gt_1,92). happensAt(inactive(wk2gt_1),92). orientation(wk2gt_2,141,92). coords(wk2gt_2,98,62,92).  visible(wk2gt_2,92). happensAt(active(wk2gt_2),92).
orientation(wk2gt_1,136,93). coords(wk2gt_1,115,53,93).  visible(wk2gt_1,93). happensAt(inactive(wk2gt_1),93). orientation(wk2gt_2,141,93). coords(wk2gt_2,98,62,93).  visible(wk2gt_2,93). happensAt(inactive(wk2gt_2),93).
orientation(wk2gt_1,137,94). coords(wk2gt_1,115,53,94).  visible(wk2gt_1,94). happensAt(inactive(wk2gt_1),94). orientation(wk2gt_2,141,94). coords(wk2gt_2,98,62,94).  visible(wk2gt_2,94). happensAt(inactive(wk2gt_2),94).
orientation(wk2gt_1,138,95). coords(wk2gt_1,115,53,95).  visible(wk2gt_1,95). happensAt(inactive(wk2gt_1),95). orientation(wk2gt_2,141,95). coords(wk2gt_2,98,62,95).  visible(wk2gt_2,95). happensAt(inactive(wk2gt_2),95).
orientation(wk2gt_1,138,96). coords(wk2gt_1,115,53,96).  visible(wk2gt_1,96). happensAt(inactive(wk2gt_1),96). orientation(wk2gt_2,141,96). coords(wk2gt_2,98,62,96).  visible(wk2gt_2,96). happensAt(inactive(wk2gt_2),96).
#show holdsAt/2.