
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

	orientation(fcgt_0,143,72). coords(fcgt_0,199,151,72).  visible(fcgt_0,72). happensAt(inactive(fcgt_0),72). orientation(fcgt_3,159,72). coords(fcgt_3,103,163,72).  visible(fcgt_3,72). happensAt(inactive(fcgt_3),72).
orientation(fcgt_0,143,73). coords(fcgt_0,199,151,73).  visible(fcgt_0,73). happensAt(inactive(fcgt_0),73). orientation(fcgt_3,159,73). coords(fcgt_3,103,163,73).  visible(fcgt_3,73). happensAt(inactive(fcgt_3),73).
orientation(fcgt_0,143,74). coords(fcgt_0,199,151,74).  visible(fcgt_0,74). happensAt(inactive(fcgt_0),74). orientation(fcgt_3,159,74). coords(fcgt_3,103,163,74).  visible(fcgt_3,74). happensAt(inactive(fcgt_3),74).
orientation(fcgt_0,143,75). coords(fcgt_0,200,151,75).  visible(fcgt_0,75). happensAt(inactive(fcgt_0),75). orientation(fcgt_3,159,75). coords(fcgt_3,103,163,75).  visible(fcgt_3,75). happensAt(inactive(fcgt_3),75).
orientation(fcgt_0,143,76). coords(fcgt_0,200,151,76).  visible(fcgt_0,76). happensAt(inactive(fcgt_0),76). orientation(fcgt_3,159,76). coords(fcgt_3,103,163,76).  visible(fcgt_3,76). happensAt(inactive(fcgt_3),76).
orientation(fcgt_0,143,77). coords(fcgt_0,200,151,77).  visible(fcgt_0,77). happensAt(inactive(fcgt_0),77). orientation(fcgt_3,159,77). coords(fcgt_3,103,163,77).  visible(fcgt_3,77). happensAt(inactive(fcgt_3),77).
orientation(fcgt_0,143,78). coords(fcgt_0,200,151,78).  visible(fcgt_0,78). happensAt(inactive(fcgt_0),78). orientation(fcgt_3,159,78). coords(fcgt_3,103,163,78).  visible(fcgt_3,78). happensAt(inactive(fcgt_3),78).
orientation(fcgt_0,143,79). coords(fcgt_0,200,151,79).  visible(fcgt_0,79). happensAt(inactive(fcgt_0),79). orientation(fcgt_3,159,79). coords(fcgt_3,103,163,79).  visible(fcgt_3,79). happensAt(inactive(fcgt_3),79).
orientation(fcgt_0,143,80). coords(fcgt_0,200,152,80).  visible(fcgt_0,80). happensAt(inactive(fcgt_0),80). orientation(fcgt_3,159,80). coords(fcgt_3,103,163,80).  visible(fcgt_3,80). happensAt(inactive(fcgt_3),80).
orientation(fcgt_0,143,81). coords(fcgt_0,200,152,81).  visible(fcgt_0,81). happensAt(inactive(fcgt_0),81). orientation(fcgt_3,159,81). coords(fcgt_3,103,163,81).  visible(fcgt_3,81). happensAt(inactive(fcgt_3),81).
orientation(fcgt_0,143,82). coords(fcgt_0,201,153,82).  visible(fcgt_0,82). happensAt(inactive(fcgt_0),82). orientation(fcgt_3,159,82). coords(fcgt_3,103,163,82).  visible(fcgt_3,82). happensAt(inactive(fcgt_3),82).
orientation(fcgt_0,147,83). coords(fcgt_0,201,153,83).  visible(fcgt_0,83). happensAt(walking(fcgt_0),83). orientation(fcgt_3,159,83). coords(fcgt_3,103,163,83).  visible(fcgt_3,83). happensAt(inactive(fcgt_3),83).
orientation(fcgt_0,147,84). coords(fcgt_0,202,154,84).  visible(fcgt_0,84). happensAt(walking(fcgt_0),84). orientation(fcgt_3,159,84). coords(fcgt_3,103,163,84).  visible(fcgt_3,84). happensAt(inactive(fcgt_3),84).
orientation(fcgt_0,147,85). coords(fcgt_0,203,154,85).  visible(fcgt_0,85). happensAt(walking(fcgt_0),85). orientation(fcgt_3,159,85). coords(fcgt_3,103,163,85).  visible(fcgt_3,85). happensAt(inactive(fcgt_3),85).
orientation(fcgt_0,147,86). coords(fcgt_0,204,154,86).  visible(fcgt_0,86). happensAt(walking(fcgt_0),86). orientation(fcgt_3,159,86). coords(fcgt_3,103,163,86).  visible(fcgt_3,86). happensAt(inactive(fcgt_3),86).
orientation(fcgt_0,151,87). coords(fcgt_0,207,155,87).  visible(fcgt_0,87). happensAt(walking(fcgt_0),87). orientation(fcgt_3,159,87). coords(fcgt_3,103,163,87).  visible(fcgt_3,87). happensAt(inactive(fcgt_3),87).
orientation(fcgt_0,151,88). coords(fcgt_0,211,158,88).  visible(fcgt_0,88). happensAt(walking(fcgt_0),88). orientation(fcgt_3,159,88). coords(fcgt_3,103,163,88).  visible(fcgt_3,88). happensAt(inactive(fcgt_3),88).
orientation(fcgt_0,151,89). coords(fcgt_0,213,160,89).  visible(fcgt_0,89). happensAt(walking(fcgt_0),89). orientation(fcgt_3,159,89). coords(fcgt_3,103,163,89).  visible(fcgt_3,89). happensAt(inactive(fcgt_3),89).
orientation(fcgt_0,151,90). coords(fcgt_0,214,162,90).  visible(fcgt_0,90). happensAt(walking(fcgt_0),90). orientation(fcgt_3,159,90). coords(fcgt_3,103,163,90).  visible(fcgt_3,90). happensAt(inactive(fcgt_3),90).
orientation(fcgt_0,151,91). coords(fcgt_0,216,165,91).  visible(fcgt_0,91). happensAt(walking(fcgt_0),91). orientation(fcgt_3,159,91). coords(fcgt_3,103,163,91).  visible(fcgt_3,91). happensAt(inactive(fcgt_3),91).
orientation(fcgt_0,151,92). coords(fcgt_0,216,167,92).  visible(fcgt_0,92). happensAt(walking(fcgt_0),92). orientation(fcgt_3,159,92). coords(fcgt_3,103,163,92).  visible(fcgt_3,92). happensAt(inactive(fcgt_3),92).
orientation(fcgt_0,151,93). coords(fcgt_0,218,169,93).  visible(fcgt_0,93). happensAt(walking(fcgt_0),93). orientation(fcgt_3,159,93). coords(fcgt_3,103,163,93).  visible(fcgt_3,93). happensAt(inactive(fcgt_3),93).
orientation(fcgt_0,151,94). coords(fcgt_0,219,172,94).  visible(fcgt_0,94). happensAt(walking(fcgt_0),94). orientation(fcgt_3,159,94). coords(fcgt_3,103,163,94).  visible(fcgt_3,94). happensAt(inactive(fcgt_3),94).
orientation(fcgt_0,151,95). coords(fcgt_0,223,174,95).  visible(fcgt_0,95). happensAt(walking(fcgt_0),95). orientation(fcgt_3,159,95). coords(fcgt_3,103,163,95).  visible(fcgt_3,95). happensAt(inactive(fcgt_3),95).
orientation(fcgt_0,154,96). coords(fcgt_0,226,177,96).  visible(fcgt_0,96). happensAt(walking(fcgt_0),96). orientation(fcgt_3,159,96). coords(fcgt_3,103,163,96).  visible(fcgt_3,96). happensAt(inactive(fcgt_3),96).
#show holdsAt/2.