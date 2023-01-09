
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

	orientation(rsfgt_1,170,12). coords(rsfgt_1,206,187,12).  visible(rsfgt_1,12). happensAt(walking(rsfgt_1),12). orientation(rsfgt_2,170,12). coords(rsfgt_2,162,215,12).  visible(rsfgt_2,12). happensAt(walking(rsfgt_2),12).
orientation(rsfgt_1,170,13). coords(rsfgt_1,206,187,13).  visible(rsfgt_1,13). happensAt(walking(rsfgt_1),13). orientation(rsfgt_2,170,13). coords(rsfgt_2,165,215,13).  visible(rsfgt_2,13). happensAt(walking(rsfgt_2),13).
orientation(rsfgt_1,170,14). coords(rsfgt_1,206,185,14).  visible(rsfgt_1,14). happensAt(walking(rsfgt_1),14). orientation(rsfgt_2,170,14). coords(rsfgt_2,167,215,14).  visible(rsfgt_2,14). happensAt(walking(rsfgt_2),14).
orientation(rsfgt_1,170,15). coords(rsfgt_1,205,183,15).  visible(rsfgt_1,15). happensAt(walking(rsfgt_1),15). orientation(rsfgt_2,170,15). coords(rsfgt_2,167,215,15).  visible(rsfgt_2,15). happensAt(walking(rsfgt_2),15).
orientation(rsfgt_1,170,16). coords(rsfgt_1,204,183,16).  visible(rsfgt_1,16). happensAt(walking(rsfgt_1),16). orientation(rsfgt_2,170,16). coords(rsfgt_2,168,215,16).  visible(rsfgt_2,16). happensAt(walking(rsfgt_2),16).
orientation(rsfgt_1,170,17). coords(rsfgt_1,204,181,17).  visible(rsfgt_1,17). happensAt(walking(rsfgt_1),17). orientation(rsfgt_2,170,17). coords(rsfgt_2,170,215,17).  visible(rsfgt_2,17). happensAt(walking(rsfgt_2),17).
orientation(rsfgt_1,170,18). coords(rsfgt_1,203,180,18).  visible(rsfgt_1,18). happensAt(walking(rsfgt_1),18). orientation(rsfgt_2,170,18). coords(rsfgt_2,171,215,18).  visible(rsfgt_2,18). happensAt(walking(rsfgt_2),18).
orientation(rsfgt_1,170,19). coords(rsfgt_1,202,179,19).  visible(rsfgt_1,19). happensAt(walking(rsfgt_1),19). orientation(rsfgt_2,170,19). coords(rsfgt_2,171,215,19).  visible(rsfgt_2,19). happensAt(walking(rsfgt_2),19).
orientation(rsfgt_1,170,20). coords(rsfgt_1,201,178,20).  visible(rsfgt_1,20). happensAt(walking(rsfgt_1),20). orientation(rsfgt_2,170,20). coords(rsfgt_2,172,215,20).  visible(rsfgt_2,20). happensAt(walking(rsfgt_2),20).
orientation(rsfgt_1,170,21). coords(rsfgt_1,198,175,21).  visible(rsfgt_1,21). happensAt(walking(rsfgt_1),21). orientation(rsfgt_2,170,21). coords(rsfgt_2,172,215,21).  visible(rsfgt_2,21). happensAt(walking(rsfgt_2),21).
orientation(rsfgt_1,170,22). coords(rsfgt_1,197,174,22).  visible(rsfgt_1,22). happensAt(walking(rsfgt_1),22). orientation(rsfgt_2,170,22). coords(rsfgt_2,173,215,22).  visible(rsfgt_2,22). happensAt(walking(rsfgt_2),22).
orientation(rsfgt_1,170,23). coords(rsfgt_1,196,172,23).  visible(rsfgt_1,23). happensAt(walking(rsfgt_1),23). orientation(rsfgt_2,170,23). coords(rsfgt_2,174,215,23).  visible(rsfgt_2,23). happensAt(walking(rsfgt_2),23).
orientation(rsfgt_1,170,24). coords(rsfgt_1,192,167,24).  visible(rsfgt_1,24). happensAt(walking(rsfgt_1),24). orientation(rsfgt_2,170,24). coords(rsfgt_2,180,217,24).  visible(rsfgt_2,24). happensAt(walking(rsfgt_2),24).
orientation(rsfgt_1,170,25). coords(rsfgt_1,191,166,25).  visible(rsfgt_1,25). happensAt(walking(rsfgt_1),25). orientation(rsfgt_2,170,25). coords(rsfgt_2,180,220,25).  visible(rsfgt_2,25). happensAt(walking(rsfgt_2),25).
orientation(rsfgt_1,170,26). coords(rsfgt_1,191,165,26).  visible(rsfgt_1,26). happensAt(walking(rsfgt_1),26). orientation(rsfgt_2,170,26). coords(rsfgt_2,183,221,26).  visible(rsfgt_2,26). happensAt(walking(rsfgt_2),26).
orientation(rsfgt_1,170,27). coords(rsfgt_1,189,164,27).  visible(rsfgt_1,27). happensAt(walking(rsfgt_1),27). orientation(rsfgt_2,170,27). coords(rsfgt_2,186,221,27).  visible(rsfgt_2,27). happensAt(walking(rsfgt_2),27).
orientation(rsfgt_1,170,28). coords(rsfgt_1,189,163,28).  visible(rsfgt_1,28). happensAt(walking(rsfgt_1),28). orientation(rsfgt_2,170,28). coords(rsfgt_2,188,222,28).  visible(rsfgt_2,28). happensAt(walking(rsfgt_2),28).
orientation(rsfgt_1,170,29). coords(rsfgt_1,189,162,29).  visible(rsfgt_1,29). happensAt(walking(rsfgt_1),29). orientation(rsfgt_2,170,29). coords(rsfgt_2,189,222,29).  visible(rsfgt_2,29). happensAt(walking(rsfgt_2),29).
orientation(rsfgt_1,170,30). coords(rsfgt_1,189,161,30).  visible(rsfgt_1,30). happensAt(walking(rsfgt_1),30). orientation(rsfgt_2,170,30). coords(rsfgt_2,189,222,30).  visible(rsfgt_2,30). happensAt(walking(rsfgt_2),30).
orientation(rsfgt_1,150,31). coords(rsfgt_1,189,160,31).  visible(rsfgt_1,31). happensAt(walking(rsfgt_1),31). orientation(rsfgt_2,170,31). coords(rsfgt_2,190,222,31).  visible(rsfgt_2,31). happensAt(walking(rsfgt_2),31).
orientation(rsfgt_1,150,32). coords(rsfgt_1,189,159,32).  visible(rsfgt_1,32). happensAt(walking(rsfgt_1),32). orientation(rsfgt_2,170,32). coords(rsfgt_2,190,222,32).  visible(rsfgt_2,32). happensAt(walking(rsfgt_2),32).
orientation(rsfgt_1,150,33). coords(rsfgt_1,188,158,33).  visible(rsfgt_1,33). happensAt(walking(rsfgt_1),33). orientation(rsfgt_2,170,33). coords(rsfgt_2,191,223,33).  visible(rsfgt_2,33). happensAt(walking(rsfgt_2),33).
orientation(rsfgt_1,150,34). coords(rsfgt_1,188,157,34).  visible(rsfgt_1,34). happensAt(walking(rsfgt_1),34). orientation(rsfgt_2,170,34). coords(rsfgt_2,193,223,34).  visible(rsfgt_2,34). happensAt(walking(rsfgt_2),34).
orientation(rsfgt_1,150,35). coords(rsfgt_1,188,155,35).  visible(rsfgt_1,35). happensAt(walking(rsfgt_1),35). orientation(rsfgt_2,170,35). coords(rsfgt_2,193,224,35).  visible(rsfgt_2,35). happensAt(walking(rsfgt_2),35).
orientation(rsfgt_1,150,36). coords(rsfgt_1,185,152,36).  visible(rsfgt_1,36). happensAt(walking(rsfgt_1),36). orientation(rsfgt_2,170,36). coords(rsfgt_2,198,226,36).  visible(rsfgt_2,36). happensAt(walking(rsfgt_2),36).
#show holdsAt/2.