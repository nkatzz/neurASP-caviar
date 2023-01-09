
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

	orientation(rsfgt_0,160,36). coords(rsfgt_0,103,127,36).  visible(rsfgt_0,36). happensAt(walking(rsfgt_0),36). orientation(rsfgt_2,170,36). coords(rsfgt_2,198,226,36).  visible(rsfgt_2,36). happensAt(walking(rsfgt_2),36).
orientation(rsfgt_0,160,37). coords(rsfgt_0,102,127,37).  visible(rsfgt_0,37). happensAt(walking(rsfgt_0),37). orientation(rsfgt_2,170,37). coords(rsfgt_2,200,228,37).  visible(rsfgt_2,37). happensAt(walking(rsfgt_2),37).
orientation(rsfgt_0,160,38). coords(rsfgt_0,102,127,38).  visible(rsfgt_0,38). happensAt(walking(rsfgt_0),38). orientation(rsfgt_2,170,38). coords(rsfgt_2,202,230,38).  visible(rsfgt_2,38). happensAt(walking(rsfgt_2),38).
orientation(rsfgt_0,160,39). coords(rsfgt_0,99,127,39).  visible(rsfgt_0,39). happensAt(walking(rsfgt_0),39). orientation(rsfgt_2,170,39). coords(rsfgt_2,203,231,39).  visible(rsfgt_2,39). happensAt(walking(rsfgt_2),39).
orientation(rsfgt_0,160,40). coords(rsfgt_0,96,128,40).  visible(rsfgt_0,40). happensAt(walking(rsfgt_0),40). orientation(rsfgt_2,170,40). coords(rsfgt_2,205,231,40).  visible(rsfgt_2,40). happensAt(walking(rsfgt_2),40).
orientation(rsfgt_0,160,41). coords(rsfgt_0,94,129,41).  visible(rsfgt_0,41). happensAt(walking(rsfgt_0),41). orientation(rsfgt_2,170,41). coords(rsfgt_2,206,233,41).  visible(rsfgt_2,41). happensAt(walking(rsfgt_2),41).
orientation(rsfgt_0,160,42). coords(rsfgt_0,93,131,42).  visible(rsfgt_0,42). happensAt(walking(rsfgt_0),42). orientation(rsfgt_2,170,42). coords(rsfgt_2,208,233,42).  visible(rsfgt_2,42). happensAt(walking(rsfgt_2),42).
orientation(rsfgt_0,160,43). coords(rsfgt_0,93,131,43).  visible(rsfgt_0,43). happensAt(walking(rsfgt_0),43). orientation(rsfgt_2,170,43). coords(rsfgt_2,208,236,43).  visible(rsfgt_2,43). happensAt(walking(rsfgt_2),43).
orientation(rsfgt_0,160,44). coords(rsfgt_0,92,131,44).  visible(rsfgt_0,44). happensAt(walking(rsfgt_0),44). orientation(rsfgt_2,170,44). coords(rsfgt_2,209,238,44).  visible(rsfgt_2,44). happensAt(walking(rsfgt_2),44).
orientation(rsfgt_0,160,45). coords(rsfgt_0,91,131,45).  visible(rsfgt_0,45). happensAt(walking(rsfgt_0),45). orientation(rsfgt_2,170,45). coords(rsfgt_2,211,240,45).  visible(rsfgt_2,45). happensAt(walking(rsfgt_2),45).
orientation(rsfgt_0,160,46). coords(rsfgt_0,92,132,46).  visible(rsfgt_0,46). happensAt(walking(rsfgt_0),46). orientation(rsfgt_2,170,46). coords(rsfgt_2,211,243,46).  visible(rsfgt_2,46). happensAt(walking(rsfgt_2),46).
orientation(rsfgt_0,160,47). coords(rsfgt_0,91,132,47).  visible(rsfgt_0,47). happensAt(walking(rsfgt_0),47). orientation(rsfgt_2,170,47). coords(rsfgt_2,211,245,47).  visible(rsfgt_2,47). happensAt(walking(rsfgt_2),47).
orientation(rsfgt_0,160,48). coords(rsfgt_0,90,134,48).  visible(rsfgt_0,48). happensAt(walking(rsfgt_0),48). orientation(rsfgt_2,170,48). coords(rsfgt_2,212,247,48).  visible(rsfgt_2,48). happensAt(walking(rsfgt_2),48).
orientation(rsfgt_0,160,49). coords(rsfgt_0,90,134,49).  visible(rsfgt_0,49). happensAt(walking(rsfgt_0),49). orientation(rsfgt_2,170,49). coords(rsfgt_2,212,251,49).  visible(rsfgt_2,49). happensAt(walking(rsfgt_2),49).
orientation(rsfgt_0,160,50). coords(rsfgt_0,89,134,50).  visible(rsfgt_0,50). happensAt(walking(rsfgt_0),50). orientation(rsfgt_2,170,50). coords(rsfgt_2,214,252,50).  visible(rsfgt_2,50). happensAt(walking(rsfgt_2),50).
orientation(rsfgt_0,160,51). coords(rsfgt_0,87,134,51).  visible(rsfgt_0,51). happensAt(walking(rsfgt_0),51). orientation(rsfgt_2,170,51). coords(rsfgt_2,216,254,51).  visible(rsfgt_2,51). happensAt(walking(rsfgt_2),51).
orientation(rsfgt_0,160,52). coords(rsfgt_0,85,134,52).  visible(rsfgt_0,52). happensAt(walking(rsfgt_0),52). orientation(rsfgt_2,170,52). coords(rsfgt_2,216,255,52).  visible(rsfgt_2,52). happensAt(walking(rsfgt_2),52).
orientation(rsfgt_0,160,53). coords(rsfgt_0,82,134,53).  visible(rsfgt_0,53). happensAt(walking(rsfgt_0),53). orientation(rsfgt_2,170,53). coords(rsfgt_2,216,255,53).  visible(rsfgt_2,53). happensAt(walking(rsfgt_2),53).
orientation(rsfgt_0,160,54). coords(rsfgt_0,82,134,54).  visible(rsfgt_0,54). happensAt(walking(rsfgt_0),54). orientation(rsfgt_2,170,54). coords(rsfgt_2,216,255,54).  visible(rsfgt_2,54). happensAt(walking(rsfgt_2),54).
orientation(rsfgt_0,160,55). coords(rsfgt_0,81,134,55).  visible(rsfgt_0,55). happensAt(walking(rsfgt_0),55). orientation(rsfgt_2,170,55). coords(rsfgt_2,216,258,55).  visible(rsfgt_2,55). happensAt(walking(rsfgt_2),55).
orientation(rsfgt_0,160,56). coords(rsfgt_0,80,134,56).  visible(rsfgt_0,56). happensAt(walking(rsfgt_0),56). orientation(rsfgt_2,170,56). coords(rsfgt_2,218,261,56).  visible(rsfgt_2,56). happensAt(walking(rsfgt_2),56).
orientation(rsfgt_0,160,57). coords(rsfgt_0,80,134,57).  visible(rsfgt_0,57). happensAt(walking(rsfgt_0),57). orientation(rsfgt_2,170,57). coords(rsfgt_2,219,263,57).  visible(rsfgt_2,57). happensAt(walking(rsfgt_2),57).
orientation(rsfgt_0,160,58). coords(rsfgt_0,80,134,58).  visible(rsfgt_0,58). happensAt(walking(rsfgt_0),58). orientation(rsfgt_2,170,58). coords(rsfgt_2,219,266,58).  visible(rsfgt_2,58). happensAt(walking(rsfgt_2),58).
orientation(rsfgt_0,160,59). coords(rsfgt_0,80,136,59).  visible(rsfgt_0,59). happensAt(walking(rsfgt_0),59). orientation(rsfgt_2,170,59). coords(rsfgt_2,220,269,59).  visible(rsfgt_2,59). happensAt(walking(rsfgt_2),59).
orientation(rsfgt_0,160,60). coords(rsfgt_0,79,136,60).  visible(rsfgt_0,60). happensAt(walking(rsfgt_0),60). orientation(rsfgt_2,170,60). coords(rsfgt_2,221,271,60).  visible(rsfgt_2,60). happensAt(walking(rsfgt_2),60).
#show holdsAt/2.