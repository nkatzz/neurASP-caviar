
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

	orientation(mws1gt_1,160,73). coords(mws1gt_1,74,187,73).  visible(mws1gt_1,73). happensAt(walking(mws1gt_1),73). orientation(mws1gt_2,162,73). coords(mws1gt_2,37,167,73).  visible(mws1gt_2,73). happensAt(walking(mws1gt_2),73).
orientation(mws1gt_1,160,74). coords(mws1gt_1,75,187,74).  visible(mws1gt_1,74). happensAt(walking(mws1gt_1),74). orientation(mws1gt_2,162,74). coords(mws1gt_2,37,168,74).  visible(mws1gt_2,74). happensAt(walking(mws1gt_2),74).
orientation(mws1gt_1,160,75). coords(mws1gt_1,77,187,75).  visible(mws1gt_1,75). happensAt(walking(mws1gt_1),75). orientation(mws1gt_2,162,75). coords(mws1gt_2,37,169,75).  visible(mws1gt_2,75). happensAt(walking(mws1gt_2),75).
orientation(mws1gt_1,160,76). coords(mws1gt_1,79,187,76).  visible(mws1gt_1,76). happensAt(walking(mws1gt_1),76). orientation(mws1gt_2,162,76). coords(mws1gt_2,37,170,76).  visible(mws1gt_2,76). happensAt(walking(mws1gt_2),76).
orientation(mws1gt_1,160,77). coords(mws1gt_1,79,186,77).  visible(mws1gt_1,77). happensAt(walking(mws1gt_1),77). orientation(mws1gt_2,162,77). coords(mws1gt_2,37,171,77).  visible(mws1gt_2,77). happensAt(walking(mws1gt_2),77).
orientation(mws1gt_1,160,78). coords(mws1gt_1,80,186,78).  visible(mws1gt_1,78). happensAt(walking(mws1gt_1),78). orientation(mws1gt_2,162,78). coords(mws1gt_2,36,172,78).  visible(mws1gt_2,78). happensAt(walking(mws1gt_2),78).
orientation(mws1gt_1,160,79). coords(mws1gt_1,81,185,79).  visible(mws1gt_1,79). happensAt(walking(mws1gt_1),79). orientation(mws1gt_2,162,79). coords(mws1gt_2,36,172,79).  visible(mws1gt_2,79). happensAt(walking(mws1gt_2),79).
orientation(mws1gt_1,160,80). coords(mws1gt_1,81,184,80).  visible(mws1gt_1,80). happensAt(walking(mws1gt_1),80). orientation(mws1gt_2,162,80). coords(mws1gt_2,36,173,80).  visible(mws1gt_2,80). happensAt(walking(mws1gt_2),80).
orientation(mws1gt_1,160,81). coords(mws1gt_1,81,184,81).  visible(mws1gt_1,81). happensAt(walking(mws1gt_1),81). orientation(mws1gt_2,162,81). coords(mws1gt_2,35,174,81).  visible(mws1gt_2,81). happensAt(walking(mws1gt_2),81).
orientation(mws1gt_1,160,82). coords(mws1gt_1,81,184,82).  visible(mws1gt_1,82). happensAt(walking(mws1gt_1),82). orientation(mws1gt_2,162,82). coords(mws1gt_2,35,175,82).  visible(mws1gt_2,82). happensAt(walking(mws1gt_2),82).
orientation(mws1gt_1,160,83). coords(mws1gt_1,81,184,83).  visible(mws1gt_1,83). happensAt(walking(mws1gt_1),83). orientation(mws1gt_2,162,83). coords(mws1gt_2,35,176,83).  visible(mws1gt_2,83). happensAt(walking(mws1gt_2),83).
orientation(mws1gt_1,160,84). coords(mws1gt_1,81,184,84).  visible(mws1gt_1,84). happensAt(walking(mws1gt_1),84). orientation(mws1gt_2,162,84). coords(mws1gt_2,35,177,84).  visible(mws1gt_2,84). happensAt(walking(mws1gt_2),84).
orientation(mws1gt_1,160,85). coords(mws1gt_1,81,183,85).  visible(mws1gt_1,85). happensAt(walking(mws1gt_1),85). orientation(mws1gt_2,162,85). coords(mws1gt_2,35,178,85).  visible(mws1gt_2,85). happensAt(walking(mws1gt_2),85).
orientation(mws1gt_1,160,86). coords(mws1gt_1,81,183,86).  visible(mws1gt_1,86). happensAt(walking(mws1gt_1),86). orientation(mws1gt_2,162,86). coords(mws1gt_2,35,179,86).  visible(mws1gt_2,86). happensAt(walking(mws1gt_2),86).
orientation(mws1gt_1,160,87). coords(mws1gt_1,82,182,87).  visible(mws1gt_1,87). happensAt(walking(mws1gt_1),87). orientation(mws1gt_2,162,87). coords(mws1gt_2,35,181,87).  visible(mws1gt_2,87). happensAt(walking(mws1gt_2),87).
orientation(mws1gt_1,160,88). coords(mws1gt_1,82,182,88).  visible(mws1gt_1,88). happensAt(walking(mws1gt_1),88). orientation(mws1gt_2,162,88). coords(mws1gt_2,35,181,88).  visible(mws1gt_2,88). happensAt(walking(mws1gt_2),88).
orientation(mws1gt_1,160,89). coords(mws1gt_1,85,182,89).  visible(mws1gt_1,89). happensAt(walking(mws1gt_1),89). orientation(mws1gt_2,162,89). coords(mws1gt_2,35,181,89).  visible(mws1gt_2,89). happensAt(walking(mws1gt_2),89).
orientation(mws1gt_1,160,90). coords(mws1gt_1,85,182,90).  visible(mws1gt_1,90). happensAt(walking(mws1gt_1),90). orientation(mws1gt_2,162,90). coords(mws1gt_2,34,181,90).  visible(mws1gt_2,90). happensAt(walking(mws1gt_2),90).
orientation(mws1gt_1,160,91). coords(mws1gt_1,86,181,91).  visible(mws1gt_1,91). happensAt(walking(mws1gt_1),91). orientation(mws1gt_2,162,91). coords(mws1gt_2,34,182,91).  visible(mws1gt_2,91). happensAt(walking(mws1gt_2),91).
orientation(mws1gt_1,160,92). coords(mws1gt_1,88,181,92).  visible(mws1gt_1,92). happensAt(walking(mws1gt_1),92). orientation(mws1gt_2,162,92). coords(mws1gt_2,34,182,92).  visible(mws1gt_2,92). happensAt(walking(mws1gt_2),92).
orientation(mws1gt_1,160,93). coords(mws1gt_1,89,180,93).  visible(mws1gt_1,93). happensAt(walking(mws1gt_1),93). orientation(mws1gt_2,162,93). coords(mws1gt_2,34,183,93).  visible(mws1gt_2,93). happensAt(walking(mws1gt_2),93).
orientation(mws1gt_1,160,94). coords(mws1gt_1,90,180,94).  visible(mws1gt_1,94). happensAt(walking(mws1gt_1),94). orientation(mws1gt_2,162,94). coords(mws1gt_2,34,185,94).  visible(mws1gt_2,94). happensAt(walking(mws1gt_2),94).
orientation(mws1gt_1,160,95). coords(mws1gt_1,90,179,95).  visible(mws1gt_1,95). happensAt(walking(mws1gt_1),95). orientation(mws1gt_2,162,95). coords(mws1gt_2,34,185,95).  visible(mws1gt_2,95). happensAt(walking(mws1gt_2),95).
orientation(mws1gt_1,160,96). coords(mws1gt_1,90,178,96).  visible(mws1gt_1,96). happensAt(walking(mws1gt_1),96). orientation(mws1gt_2,162,96). coords(mws1gt_2,34,187,96).  visible(mws1gt_2,96). happensAt(walking(mws1gt_2),96).
orientation(mws1gt_1,160,97). coords(mws1gt_1,91,177,97).  visible(mws1gt_1,97). happensAt(walking(mws1gt_1),97). orientation(mws1gt_2,162,97). coords(mws1gt_2,34,188,97).  visible(mws1gt_2,97). happensAt(walking(mws1gt_2),97).
#show holdsAt/2.