
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

	orientation(rsfgt_0,160,72). coords(rsfgt_0,70,141,72).  visible(rsfgt_0,72). happensAt(walking(rsfgt_0),72). orientation(rsfgt_1,150,72). coords(rsfgt_1,170,116,72).  visible(rsfgt_1,72). happensAt(walking(rsfgt_1),72).
orientation(rsfgt_0,160,73). coords(rsfgt_0,70,142,73).  visible(rsfgt_0,73). happensAt(walking(rsfgt_0),73). orientation(rsfgt_1,150,73). coords(rsfgt_1,170,116,73).  visible(rsfgt_1,73). happensAt(walking(rsfgt_1),73).
orientation(rsfgt_0,160,74). coords(rsfgt_0,69,143,74).  visible(rsfgt_0,74). happensAt(walking(rsfgt_0),74). orientation(rsfgt_1,150,74). coords(rsfgt_1,170,116,74).  visible(rsfgt_1,74). happensAt(walking(rsfgt_1),74).
orientation(rsfgt_0,160,75). coords(rsfgt_0,68,143,75).  visible(rsfgt_0,75). happensAt(walking(rsfgt_0),75). orientation(rsfgt_1,150,75). coords(rsfgt_1,170,115,75).  visible(rsfgt_1,75). happensAt(walking(rsfgt_1),75).
orientation(rsfgt_0,160,76). coords(rsfgt_0,67,143,76).  visible(rsfgt_0,76). happensAt(walking(rsfgt_0),76). orientation(rsfgt_1,150,76). coords(rsfgt_1,170,115,76).  visible(rsfgt_1,76). happensAt(walking(rsfgt_1),76).
orientation(rsfgt_0,160,77). coords(rsfgt_0,66,143,77).  visible(rsfgt_0,77). happensAt(walking(rsfgt_0),77). orientation(rsfgt_1,150,77). coords(rsfgt_1,168,113,77).  visible(rsfgt_1,77). happensAt(walking(rsfgt_1),77).
orientation(rsfgt_0,160,78). coords(rsfgt_0,65,143,78).  visible(rsfgt_0,78). happensAt(walking(rsfgt_0),78). orientation(rsfgt_1,150,78). coords(rsfgt_1,168,113,78).  visible(rsfgt_1,78). happensAt(walking(rsfgt_1),78).
orientation(rsfgt_0,160,79). coords(rsfgt_0,64,143,79).  visible(rsfgt_0,79). happensAt(walking(rsfgt_0),79). orientation(rsfgt_1,150,79). coords(rsfgt_1,168,113,79).  visible(rsfgt_1,79). happensAt(walking(rsfgt_1),79).
orientation(rsfgt_0,160,80). coords(rsfgt_0,62,143,80).  visible(rsfgt_0,80). happensAt(walking(rsfgt_0),80). orientation(rsfgt_1,150,80). coords(rsfgt_1,168,113,80).  visible(rsfgt_1,80). happensAt(walking(rsfgt_1),80).
orientation(rsfgt_0,160,81). coords(rsfgt_0,62,143,81).  visible(rsfgt_0,81). happensAt(walking(rsfgt_0),81). orientation(rsfgt_1,150,81). coords(rsfgt_1,168,113,81).  visible(rsfgt_1,81). happensAt(walking(rsfgt_1),81).
orientation(rsfgt_0,160,82). coords(rsfgt_0,62,144,82).  visible(rsfgt_0,82). happensAt(walking(rsfgt_0),82). orientation(rsfgt_1,150,82). coords(rsfgt_1,168,113,82).  visible(rsfgt_1,82). happensAt(walking(rsfgt_1),82).
orientation(rsfgt_0,160,83). coords(rsfgt_0,62,144,83).  visible(rsfgt_0,83). happensAt(walking(rsfgt_0),83). orientation(rsfgt_1,150,83). coords(rsfgt_1,168,112,83).  visible(rsfgt_1,83). happensAt(walking(rsfgt_1),83).
orientation(rsfgt_0,160,84). coords(rsfgt_0,61,144,84).  visible(rsfgt_0,84). happensAt(walking(rsfgt_0),84). orientation(rsfgt_1,150,84). coords(rsfgt_1,166,112,84).  visible(rsfgt_1,84). happensAt(walking(rsfgt_1),84).
orientation(rsfgt_0,160,85). coords(rsfgt_0,61,144,85).  visible(rsfgt_0,85). happensAt(walking(rsfgt_0),85). orientation(rsfgt_1,150,85). coords(rsfgt_1,164,112,85).  visible(rsfgt_1,85). happensAt(walking(rsfgt_1),85).
orientation(rsfgt_0,160,86). coords(rsfgt_0,60,144,86).  visible(rsfgt_0,86). happensAt(walking(rsfgt_0),86). orientation(rsfgt_1,150,86). coords(rsfgt_1,162,112,86).  visible(rsfgt_1,86). happensAt(walking(rsfgt_1),86).
orientation(rsfgt_0,160,87). coords(rsfgt_0,60,146,87).  visible(rsfgt_0,87). happensAt(walking(rsfgt_0),87). orientation(rsfgt_1,150,87). coords(rsfgt_1,161,110,87).  visible(rsfgt_1,87). happensAt(walking(rsfgt_1),87).
orientation(rsfgt_0,160,88). coords(rsfgt_0,60,146,88).  visible(rsfgt_0,88). happensAt(walking(rsfgt_0),88). orientation(rsfgt_1,150,88). coords(rsfgt_1,161,110,88).  visible(rsfgt_1,88). happensAt(walking(rsfgt_1),88).
orientation(rsfgt_0,160,89). coords(rsfgt_0,58,146,89).  visible(rsfgt_0,89). happensAt(walking(rsfgt_0),89). orientation(rsfgt_1,150,89). coords(rsfgt_1,161,108,89).  visible(rsfgt_1,89). happensAt(walking(rsfgt_1),89).
orientation(rsfgt_0,160,90). coords(rsfgt_0,58,146,90).  visible(rsfgt_0,90). happensAt(walking(rsfgt_0),90). orientation(rsfgt_1,150,90). coords(rsfgt_1,161,107,90).  visible(rsfgt_1,90). happensAt(walking(rsfgt_1),90).
orientation(rsfgt_0,160,91). coords(rsfgt_0,58,147,91).  visible(rsfgt_0,91). happensAt(walking(rsfgt_0),91). orientation(rsfgt_1,150,91). coords(rsfgt_1,161,106,91).  visible(rsfgt_1,91). happensAt(active(rsfgt_1),91).
orientation(rsfgt_0,160,92). coords(rsfgt_0,57,147,92).  visible(rsfgt_0,92). happensAt(walking(rsfgt_0),92). orientation(rsfgt_1,150,92). coords(rsfgt_1,161,106,92).  visible(rsfgt_1,92). happensAt(active(rsfgt_1),92).
orientation(rsfgt_0,160,93). coords(rsfgt_0,56,148,93).  visible(rsfgt_0,93). happensAt(walking(rsfgt_0),93). orientation(rsfgt_1,150,93). coords(rsfgt_1,161,106,93).  visible(rsfgt_1,93). happensAt(active(rsfgt_1),93).
orientation(rsfgt_0,160,94). coords(rsfgt_0,56,148,94).  visible(rsfgt_0,94). happensAt(walking(rsfgt_0),94). orientation(rsfgt_1,150,94). coords(rsfgt_1,165,104,94).  visible(rsfgt_1,94). happensAt(active(rsfgt_1),94).
orientation(rsfgt_0,160,95). coords(rsfgt_0,56,149,95).  visible(rsfgt_0,95). happensAt(walking(rsfgt_0),95). orientation(rsfgt_1,150,95). coords(rsfgt_1,165,104,95).  visible(rsfgt_1,95). happensAt(active(rsfgt_1),95).
orientation(rsfgt_0,160,96). coords(rsfgt_0,56,150,96).  visible(rsfgt_0,96). happensAt(walking(rsfgt_0),96). orientation(rsfgt_1,150,96). coords(rsfgt_1,165,103,96).  visible(rsfgt_1,96). happensAt(active(rsfgt_1),96).
#show holdsAt/2.