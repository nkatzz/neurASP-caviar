
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

	orientation(ricgt_4,172,71). coords(ricgt_4,27,131,71).  visible(ricgt_4,71). happensAt(walking(ricgt_4),71). orientation(ricgt_5,0,71). coords(ricgt_5,28,123,71).  visible(ricgt_5,71). happensAt(walking(ricgt_5),71).
orientation(ricgt_4,172,72). coords(ricgt_4,27,132,72).  visible(ricgt_4,72). happensAt(walking(ricgt_4),72). orientation(ricgt_5,0,72). coords(ricgt_5,28,123,72).  visible(ricgt_5,72). happensAt(walking(ricgt_5),72).
orientation(ricgt_4,172,73). coords(ricgt_4,27,132,73).  visible(ricgt_4,73). happensAt(walking(ricgt_4),73). orientation(ricgt_5,0,73). coords(ricgt_5,28,124,73).  visible(ricgt_5,73). happensAt(walking(ricgt_5),73).
orientation(ricgt_4,172,74). coords(ricgt_4,27,132,74).  visible(ricgt_4,74). happensAt(walking(ricgt_4),74). orientation(ricgt_5,0,74). coords(ricgt_5,28,124,74).  visible(ricgt_5,74). happensAt(walking(ricgt_5),74).
orientation(ricgt_4,172,75). coords(ricgt_4,27,132,75).  visible(ricgt_4,75). happensAt(walking(ricgt_4),75). orientation(ricgt_5,0,75). coords(ricgt_5,28,124,75).  visible(ricgt_5,75). happensAt(walking(ricgt_5),75).
orientation(ricgt_4,172,76). coords(ricgt_4,27,132,76).  visible(ricgt_4,76). happensAt(walking(ricgt_4),76). orientation(ricgt_5,0,76). coords(ricgt_5,28,124,76).  visible(ricgt_5,76). happensAt(walking(ricgt_5),76).
orientation(ricgt_4,172,77). coords(ricgt_4,27,132,77).  visible(ricgt_4,77). happensAt(walking(ricgt_4),77). orientation(ricgt_5,0,77). coords(ricgt_5,28,124,77).  visible(ricgt_5,77). happensAt(walking(ricgt_5),77).
orientation(ricgt_4,172,78). coords(ricgt_4,27,132,78).  visible(ricgt_4,78). happensAt(walking(ricgt_4),78). orientation(ricgt_5,0,78). coords(ricgt_5,28,124,78).  visible(ricgt_5,78). happensAt(walking(ricgt_5),78).
orientation(ricgt_4,172,79). coords(ricgt_4,27,132,79).  visible(ricgt_4,79). happensAt(walking(ricgt_4),79). orientation(ricgt_5,0,79). coords(ricgt_5,28,124,79).  visible(ricgt_5,79). happensAt(walking(ricgt_5),79).
orientation(ricgt_4,172,80). coords(ricgt_4,27,132,80).  visible(ricgt_4,80). happensAt(walking(ricgt_4),80). orientation(ricgt_5,0,80). coords(ricgt_5,28,124,80).  visible(ricgt_5,80). happensAt(walking(ricgt_5),80).
orientation(ricgt_4,172,81). coords(ricgt_4,27,133,81).  visible(ricgt_4,81). happensAt(walking(ricgt_4),81). orientation(ricgt_5,0,81). coords(ricgt_5,28,124,81).  visible(ricgt_5,81). happensAt(walking(ricgt_5),81).
orientation(ricgt_4,172,82). coords(ricgt_4,27,133,82).  visible(ricgt_4,82). happensAt(walking(ricgt_4),82). orientation(ricgt_5,0,82). coords(ricgt_5,28,124,82).  visible(ricgt_5,82). happensAt(walking(ricgt_5),82).
orientation(ricgt_4,172,83). coords(ricgt_4,27,133,83).  visible(ricgt_4,83). happensAt(walking(ricgt_4),83). orientation(ricgt_5,0,83). coords(ricgt_5,27,125,83).  visible(ricgt_5,83). happensAt(walking(ricgt_5),83).
orientation(ricgt_4,172,84). coords(ricgt_4,27,133,84).  visible(ricgt_4,84). happensAt(walking(ricgt_4),84). orientation(ricgt_5,0,84). coords(ricgt_5,27,127,84).  visible(ricgt_5,84). happensAt(walking(ricgt_5),84).
orientation(ricgt_4,172,85). coords(ricgt_4,27,133,85).  visible(ricgt_4,85). happensAt(walking(ricgt_4),85). orientation(ricgt_5,0,85). coords(ricgt_5,27,126,85).  visible(ricgt_5,85). happensAt(walking(ricgt_5),85).
orientation(ricgt_4,172,86). coords(ricgt_4,27,133,86).  visible(ricgt_4,86). happensAt(walking(ricgt_4),86). orientation(ricgt_5,0,86). coords(ricgt_5,27,126,86).  visible(ricgt_5,86). happensAt(walking(ricgt_5),86).
orientation(ricgt_4,172,87). coords(ricgt_4,27,133,87).  visible(ricgt_4,87). happensAt(walking(ricgt_4),87). orientation(ricgt_5,0,87). coords(ricgt_5,27,127,87).  visible(ricgt_5,87). happensAt(walking(ricgt_5),87).
orientation(ricgt_4,172,88). coords(ricgt_4,27,133,88).  visible(ricgt_4,88). happensAt(walking(ricgt_4),88). orientation(ricgt_5,0,88). coords(ricgt_5,27,127,88).  visible(ricgt_5,88). happensAt(walking(ricgt_5),88).
orientation(ricgt_4,172,89). coords(ricgt_4,27,133,89).  visible(ricgt_4,89). happensAt(walking(ricgt_4),89). orientation(ricgt_5,0,89). coords(ricgt_5,27,127,89).  visible(ricgt_5,89). happensAt(walking(ricgt_5),89).
orientation(ricgt_4,172,90). coords(ricgt_4,27,133,90).  visible(ricgt_4,90). happensAt(walking(ricgt_4),90). orientation(ricgt_5,0,90). coords(ricgt_5,27,127,90).  visible(ricgt_5,90). happensAt(walking(ricgt_5),90).
orientation(ricgt_4,172,91). coords(ricgt_4,27,133,91).  visible(ricgt_4,91). happensAt(walking(ricgt_4),91). orientation(ricgt_5,0,91). coords(ricgt_5,27,127,91).  visible(ricgt_5,91). happensAt(walking(ricgt_5),91).
orientation(ricgt_4,172,92). coords(ricgt_4,27,133,92).  visible(ricgt_4,92). happensAt(walking(ricgt_4),92). orientation(ricgt_5,0,92). coords(ricgt_5,27,126,92).  visible(ricgt_5,92). happensAt(walking(ricgt_5),92).
orientation(ricgt_4,172,93). coords(ricgt_4,26,133,93).  visible(ricgt_4,93). happensAt(walking(ricgt_4),93). orientation(ricgt_5,0,93). coords(ricgt_5,27,126,93).  visible(ricgt_5,93). happensAt(walking(ricgt_5),93).
orientation(ricgt_4,172,94). coords(ricgt_4,26,134,94).  visible(ricgt_4,94). happensAt(walking(ricgt_4),94). orientation(ricgt_5,0,94). coords(ricgt_5,27,126,94).  visible(ricgt_5,94). happensAt(walking(ricgt_5),94).
orientation(ricgt_4,172,95). coords(ricgt_4,26,134,95).  visible(ricgt_4,95). happensAt(walking(ricgt_4),95). orientation(ricgt_5,0,95). coords(ricgt_5,27,126,95).  visible(ricgt_5,95). happensAt(walking(ricgt_5),95).
#show holdsAt/2.