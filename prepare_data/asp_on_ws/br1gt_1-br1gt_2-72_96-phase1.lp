
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

	orientation(br1gt_1,164,72). coords(br1gt_1,97,178,72).  visible(br1gt_1,72). happensAt(walking(br1gt_1),72). orientation(br1gt_2,171,72). coords(br1gt_2,86,207,72).  visible(br1gt_2,72). happensAt(inactive(br1gt_2),72).
orientation(br1gt_1,164,73). coords(br1gt_1,96,177,73).  visible(br1gt_1,73). happensAt(walking(br1gt_1),73). orientation(br1gt_2,171,73). coords(br1gt_2,86,207,73).  visible(br1gt_2,73). happensAt(inactive(br1gt_2),73).
orientation(br1gt_1,164,74). coords(br1gt_1,95,176,74).  visible(br1gt_1,74). happensAt(walking(br1gt_1),74). orientation(br1gt_2,171,74). coords(br1gt_2,86,207,74).  visible(br1gt_2,74). happensAt(inactive(br1gt_2),74).
orientation(br1gt_1,164,75). coords(br1gt_1,95,176,75).  visible(br1gt_1,75). happensAt(walking(br1gt_1),75). orientation(br1gt_2,171,75). coords(br1gt_2,86,207,75).  visible(br1gt_2,75). happensAt(inactive(br1gt_2),75).
orientation(br1gt_1,164,76). coords(br1gt_1,95,176,76).  visible(br1gt_1,76). happensAt(walking(br1gt_1),76). orientation(br1gt_2,171,76). coords(br1gt_2,86,207,76).  visible(br1gt_2,76). happensAt(inactive(br1gt_2),76).
orientation(br1gt_1,164,77). coords(br1gt_1,94,176,77).  visible(br1gt_1,77). happensAt(walking(br1gt_1),77). orientation(br1gt_2,171,77). coords(br1gt_2,86,207,77).  visible(br1gt_2,77). happensAt(inactive(br1gt_2),77).
orientation(br1gt_1,164,78). coords(br1gt_1,94,176,78).  visible(br1gt_1,78). happensAt(walking(br1gt_1),78). orientation(br1gt_2,171,78). coords(br1gt_2,86,207,78).  visible(br1gt_2,78). happensAt(inactive(br1gt_2),78).
orientation(br1gt_1,164,79). coords(br1gt_1,93,175,79).  visible(br1gt_1,79). happensAt(walking(br1gt_1),79). orientation(br1gt_2,171,79). coords(br1gt_2,86,207,79).  visible(br1gt_2,79). happensAt(inactive(br1gt_2),79).
orientation(br1gt_1,164,80). coords(br1gt_1,93,175,80).  visible(br1gt_1,80). happensAt(walking(br1gt_1),80). orientation(br1gt_2,171,80). coords(br1gt_2,86,207,80).  visible(br1gt_2,80). happensAt(inactive(br1gt_2),80).
orientation(br1gt_1,164,81). coords(br1gt_1,92,173,81).  visible(br1gt_1,81). happensAt(walking(br1gt_1),81). orientation(br1gt_2,171,81). coords(br1gt_2,86,207,81).  visible(br1gt_2,81). happensAt(inactive(br1gt_2),81).
orientation(br1gt_1,164,82). coords(br1gt_1,90,171,82).  visible(br1gt_1,82). happensAt(walking(br1gt_1),82). orientation(br1gt_2,171,82). coords(br1gt_2,86,207,82).  visible(br1gt_2,82). happensAt(inactive(br1gt_2),82).
orientation(br1gt_1,164,83). coords(br1gt_1,88,171,83).  visible(br1gt_1,83). happensAt(walking(br1gt_1),83). orientation(br1gt_2,171,83). coords(br1gt_2,86,207,83).  visible(br1gt_2,83). happensAt(inactive(br1gt_2),83).
orientation(br1gt_1,164,84). coords(br1gt_1,87,171,84).  visible(br1gt_1,84). happensAt(walking(br1gt_1),84). orientation(br1gt_2,171,84). coords(br1gt_2,86,207,84).  visible(br1gt_2,84). happensAt(inactive(br1gt_2),84).
orientation(br1gt_1,164,85). coords(br1gt_1,85,171,85).  visible(br1gt_1,85). happensAt(walking(br1gt_1),85). orientation(br1gt_2,171,85). coords(br1gt_2,86,207,85).  visible(br1gt_2,85). happensAt(inactive(br1gt_2),85).
orientation(br1gt_1,164,86). coords(br1gt_1,84,171,86).  visible(br1gt_1,86). happensAt(walking(br1gt_1),86). orientation(br1gt_2,171,86). coords(br1gt_2,86,207,86).  visible(br1gt_2,86). happensAt(inactive(br1gt_2),86).
orientation(br1gt_1,164,87). coords(br1gt_1,84,171,87).  visible(br1gt_1,87). happensAt(walking(br1gt_1),87). orientation(br1gt_2,171,87). coords(br1gt_2,86,207,87).  visible(br1gt_2,87). happensAt(inactive(br1gt_2),87).
orientation(br1gt_1,164,88). coords(br1gt_1,83,170,88).  visible(br1gt_1,88). happensAt(walking(br1gt_1),88). orientation(br1gt_2,171,88). coords(br1gt_2,86,207,88).  visible(br1gt_2,88). happensAt(inactive(br1gt_2),88).
orientation(br1gt_1,164,89). coords(br1gt_1,83,170,89).  visible(br1gt_1,89). happensAt(walking(br1gt_1),89). orientation(br1gt_2,171,89). coords(br1gt_2,86,207,89).  visible(br1gt_2,89). happensAt(inactive(br1gt_2),89).
orientation(br1gt_1,164,90). coords(br1gt_1,83,170,90).  visible(br1gt_1,90). happensAt(walking(br1gt_1),90). orientation(br1gt_2,171,90). coords(br1gt_2,86,207,90).  visible(br1gt_2,90). happensAt(inactive(br1gt_2),90).
orientation(br1gt_1,164,91). coords(br1gt_1,83,170,91).  visible(br1gt_1,91). happensAt(walking(br1gt_1),91). orientation(br1gt_2,171,91). coords(br1gt_2,86,207,91).  visible(br1gt_2,91). happensAt(inactive(br1gt_2),91).
orientation(br1gt_1,164,92). coords(br1gt_1,83,170,92).  visible(br1gt_1,92). happensAt(walking(br1gt_1),92). orientation(br1gt_2,171,92). coords(br1gt_2,86,207,92).  visible(br1gt_2,92). happensAt(inactive(br1gt_2),92).
orientation(br1gt_1,164,93). coords(br1gt_1,81,170,93).  visible(br1gt_1,93). happensAt(walking(br1gt_1),93). orientation(br1gt_2,171,93). coords(br1gt_2,86,207,93).  visible(br1gt_2,93). happensAt(inactive(br1gt_2),93).
orientation(br1gt_1,164,94). coords(br1gt_1,80,169,94).  visible(br1gt_1,94). happensAt(walking(br1gt_1),94). orientation(br1gt_2,171,94). coords(br1gt_2,86,207,94).  visible(br1gt_2,94). happensAt(inactive(br1gt_2),94).
orientation(br1gt_1,164,95). coords(br1gt_1,78,169,95).  visible(br1gt_1,95). happensAt(walking(br1gt_1),95). orientation(br1gt_2,171,95). coords(br1gt_2,86,207,95).  visible(br1gt_2,95). happensAt(inactive(br1gt_2),95).
orientation(br1gt_1,164,96). coords(br1gt_1,77,168,96).  visible(br1gt_1,96). happensAt(walking(br1gt_1),96). orientation(br1gt_2,171,96). coords(br1gt_2,86,207,96).  visible(br1gt_2,96). happensAt(inactive(br1gt_2),96).
#show holdsAt/2.