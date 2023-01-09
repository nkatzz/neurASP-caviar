
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

	orientation(fcgt_3,159,72). coords(fcgt_3,103,163,72).  visible(fcgt_3,72). happensAt(inactive(fcgt_3),72). orientation(fcgt_4,165,72). coords(fcgt_4,100,208,72).  visible(fcgt_4,72). happensAt(inactive(fcgt_4),72).
orientation(fcgt_3,159,73). coords(fcgt_3,103,163,73).  visible(fcgt_3,73). happensAt(inactive(fcgt_3),73). orientation(fcgt_4,165,73). coords(fcgt_4,100,208,73).  visible(fcgt_4,73). happensAt(inactive(fcgt_4),73).
orientation(fcgt_3,159,74). coords(fcgt_3,103,163,74).  visible(fcgt_3,74). happensAt(inactive(fcgt_3),74). orientation(fcgt_4,165,74). coords(fcgt_4,100,208,74).  visible(fcgt_4,74). happensAt(inactive(fcgt_4),74).
orientation(fcgt_3,159,75). coords(fcgt_3,103,163,75).  visible(fcgt_3,75). happensAt(inactive(fcgt_3),75). orientation(fcgt_4,165,75). coords(fcgt_4,100,208,75).  visible(fcgt_4,75). happensAt(inactive(fcgt_4),75).
orientation(fcgt_3,159,76). coords(fcgt_3,103,163,76).  visible(fcgt_3,76). happensAt(inactive(fcgt_3),76). orientation(fcgt_4,165,76). coords(fcgt_4,100,208,76).  visible(fcgt_4,76). happensAt(inactive(fcgt_4),76).
orientation(fcgt_3,159,77). coords(fcgt_3,103,163,77).  visible(fcgt_3,77). happensAt(inactive(fcgt_3),77). orientation(fcgt_4,165,77). coords(fcgt_4,100,208,77).  visible(fcgt_4,77). happensAt(inactive(fcgt_4),77).
orientation(fcgt_3,159,78). coords(fcgt_3,103,163,78).  visible(fcgt_3,78). happensAt(inactive(fcgt_3),78). orientation(fcgt_4,165,78). coords(fcgt_4,100,209,78).  visible(fcgt_4,78). happensAt(inactive(fcgt_4),78).
orientation(fcgt_3,159,79). coords(fcgt_3,103,163,79).  visible(fcgt_3,79). happensAt(inactive(fcgt_3),79). orientation(fcgt_4,165,79). coords(fcgt_4,100,209,79).  visible(fcgt_4,79). happensAt(inactive(fcgt_4),79).
orientation(fcgt_3,159,80). coords(fcgt_3,103,163,80).  visible(fcgt_3,80). happensAt(inactive(fcgt_3),80). orientation(fcgt_4,165,80). coords(fcgt_4,100,209,80).  visible(fcgt_4,80). happensAt(inactive(fcgt_4),80).
orientation(fcgt_3,159,81). coords(fcgt_3,103,163,81).  visible(fcgt_3,81). happensAt(inactive(fcgt_3),81). orientation(fcgt_4,165,81). coords(fcgt_4,100,209,81).  visible(fcgt_4,81). happensAt(inactive(fcgt_4),81).
orientation(fcgt_3,159,82). coords(fcgt_3,103,163,82).  visible(fcgt_3,82). happensAt(inactive(fcgt_3),82). orientation(fcgt_4,165,82). coords(fcgt_4,100,209,82).  visible(fcgt_4,82). happensAt(inactive(fcgt_4),82).
orientation(fcgt_3,159,83). coords(fcgt_3,103,163,83).  visible(fcgt_3,83). happensAt(inactive(fcgt_3),83). orientation(fcgt_4,165,83). coords(fcgt_4,100,209,83).  visible(fcgt_4,83). happensAt(inactive(fcgt_4),83).
orientation(fcgt_3,159,84). coords(fcgt_3,103,163,84).  visible(fcgt_3,84). happensAt(inactive(fcgt_3),84). orientation(fcgt_4,165,84). coords(fcgt_4,100,209,84).  visible(fcgt_4,84). happensAt(inactive(fcgt_4),84).
orientation(fcgt_3,159,85). coords(fcgt_3,103,163,85).  visible(fcgt_3,85). happensAt(inactive(fcgt_3),85). orientation(fcgt_4,165,85). coords(fcgt_4,101,209,85).  visible(fcgt_4,85). happensAt(inactive(fcgt_4),85).
orientation(fcgt_3,159,86). coords(fcgt_3,103,163,86).  visible(fcgt_3,86). happensAt(inactive(fcgt_3),86). orientation(fcgt_4,165,86). coords(fcgt_4,101,209,86).  visible(fcgt_4,86). happensAt(inactive(fcgt_4),86).
orientation(fcgt_3,159,87). coords(fcgt_3,103,163,87).  visible(fcgt_3,87). happensAt(inactive(fcgt_3),87). orientation(fcgt_4,165,87). coords(fcgt_4,101,209,87).  visible(fcgt_4,87). happensAt(inactive(fcgt_4),87).
orientation(fcgt_3,159,88). coords(fcgt_3,103,163,88).  visible(fcgt_3,88). happensAt(inactive(fcgt_3),88). orientation(fcgt_4,165,88). coords(fcgt_4,101,209,88).  visible(fcgt_4,88). happensAt(inactive(fcgt_4),88).
orientation(fcgt_3,159,89). coords(fcgt_3,103,163,89).  visible(fcgt_3,89). happensAt(inactive(fcgt_3),89). orientation(fcgt_4,165,89). coords(fcgt_4,101,209,89).  visible(fcgt_4,89). happensAt(inactive(fcgt_4),89).
orientation(fcgt_3,159,90). coords(fcgt_3,103,163,90).  visible(fcgt_3,90). happensAt(inactive(fcgt_3),90). orientation(fcgt_4,165,90). coords(fcgt_4,101,209,90).  visible(fcgt_4,90). happensAt(inactive(fcgt_4),90).
orientation(fcgt_3,159,91). coords(fcgt_3,103,163,91).  visible(fcgt_3,91). happensAt(inactive(fcgt_3),91). orientation(fcgt_4,165,91). coords(fcgt_4,101,209,91).  visible(fcgt_4,91). happensAt(inactive(fcgt_4),91).
orientation(fcgt_3,159,92). coords(fcgt_3,103,163,92).  visible(fcgt_3,92). happensAt(inactive(fcgt_3),92). orientation(fcgt_4,165,92). coords(fcgt_4,101,210,92).  visible(fcgt_4,92). happensAt(inactive(fcgt_4),92).
orientation(fcgt_3,159,93). coords(fcgt_3,103,163,93).  visible(fcgt_3,93). happensAt(inactive(fcgt_3),93). orientation(fcgt_4,165,93). coords(fcgt_4,101,210,93).  visible(fcgt_4,93). happensAt(inactive(fcgt_4),93).
orientation(fcgt_3,159,94). coords(fcgt_3,103,163,94).  visible(fcgt_3,94). happensAt(inactive(fcgt_3),94). orientation(fcgt_4,165,94). coords(fcgt_4,101,210,94).  visible(fcgt_4,94). happensAt(inactive(fcgt_4),94).
orientation(fcgt_3,159,95). coords(fcgt_3,103,163,95).  visible(fcgt_3,95). happensAt(inactive(fcgt_3),95). orientation(fcgt_4,165,95). coords(fcgt_4,101,210,95).  visible(fcgt_4,95). happensAt(inactive(fcgt_4),95).
orientation(fcgt_3,159,96). coords(fcgt_3,103,163,96).  visible(fcgt_3,96). happensAt(inactive(fcgt_3),96). orientation(fcgt_4,165,96). coords(fcgt_4,101,210,96).  visible(fcgt_4,96). happensAt(inactive(fcgt_4),96).
#show holdsAt/2.